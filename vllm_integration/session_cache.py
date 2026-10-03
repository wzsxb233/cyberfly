"""Lab-owner-only inference heads, pinned after their full checkpoint checks.

This does not cache observations. Synchronous callers validate every new brain
NPZ on the owner; request_heads moves that work to each private worker. File
fingerprints detect a change to the pinned version; they are not substituted
for the initial SHA/tensor/identity validation. Training uses separate objects.
"""
from copy import copy, deepcopy
import hashlib
import json
from pathlib import Path
import threading
import time

from shared_io.sensory_sources import SharedSensorySources, BASE

ROOT = Path(__file__).resolve().parents[1]


def file_revision(path):
    """Detect replacement/in-place updates of an already validated local file."""
    from .path_identity import regular_file_revision
    return regular_file_revision(path, root=ROOT)


class OwnerSensorySources(SharedSensorySources):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.revisions = {}
        self.loads = self.hits = self.invalidations = 0
        self.load_seconds = 0.
        self.last_body = None

    def body_adapter(self, packet):
        self.owner._owner()
        from shared_io.body_senses import BodySenseCodec
        BodySenseCodec._validate_packet(packet)
        codec = next((entry[1].codec for entry in self.body.values()
                      if entry[1].codec.layout_sha == packet['model_layout_sha256']), None)
        if codec is None:
            codec = BodySenseCodec(packet)
        layout = codec.identity['feature_layout_sha256']
        pointer = BASE / 'body' / layout / 'current.json'
        if not pointer.is_file():
            # A new body must be explicitly initialized and saved before use.
            # In particular, caching cannot turn a missing version into an
            # apparently verified random head.
            raise ValueError('No saved body sensory version for the actual body layout')
        raw = pointer.read_bytes()
        saved = json.loads(raw)
        from .path_identity import resolve_path
        path = resolve_path(saved['path'], root=ROOT, strict=True)
        if not path.is_relative_to(resolve_path(BASE / 'body' / layout / 'versions', root=ROOT)):
            raise ValueError('Body sensory version is outside its exact layout directory')
        revision = (hashlib.sha256(raw).hexdigest(), file_revision(path / 'metadata.json'),
                    file_revision(path / 'weights.npz'))
        old = self.revisions.get(layout)
        if old != revision:
            if old is not None:
                self.invalidations += 1
            self.body.pop(layout, None)
        started = time.perf_counter()
        model = super().body_adapter(packet) if old != revision else self.body[layout][1]
        if old != revision:
            if pointer.read_bytes() != raw or revision[1:] != (
                    file_revision(path / 'metadata.json'), file_revision(path / 'weights.npz')):
                self.body.pop(layout, None)
                raise ValueError('Body sensory version changed during full checkpoint verification')
            model.requires_grad_(False).eval()
            self.loads += 1
            self.load_seconds += time.perf_counter() - started
            self.revisions[layout] = revision
        else:
            self.hits += 1
        self.last_body = {'layout_sha256': layout, 'pointer_sha256': revision[0],
                          'checkpoint_path': str(path), 'adapter_sha256': saved['adapter_sha256'],
                          'training_steps': model.training_steps,
                          'checkpoint_revalidated_this_access': old != revision}
        return model


class SessionInferenceCache:
    def __init__(self):
        self.owner_id = threading.get_ident()
        self.environment = None
        self.entry = None
        self.sensory_sources = OwnerSensorySources(self)
        self.loads = self.hits = self.invalidations = self.brain_checks = 0
        self.load_seconds = 0.
        self.last_reason = None
        self.request_forks = self.body_embedding_loads = 0
        self.body_embedding_revisions = {}

    def _owner(self):
        if threading.get_ident() != self.owner_id:
            raise RuntimeError('Inference cache and mutable codecs belong only to the Lab owner thread')

    def bind_environment(self, env, brain):
        self._owner()
        if self.environment is not None and (self.environment[0] is not env or self.environment[1] is not brain):
            self.clear(reason='environment_or_brain_replaced')
        self.environment = (env, brain)

    def clear(self, *, reason='explicit_clear'):
        self._owner()
        if self.entry is not None or self.sensory_sources.body:
            self.invalidations += 1
        self.entry = None
        self.sensory_sources = OwnerSensorySources(self)
        self.environment = None
        self.last_reason = reason
        self.body_embedding_revisions = {}

    def coupler(self, brain_state, *, io_scope='sensory_motor', checkpoint=None):
        return self._coupler(brain_state, io_scope=io_scope, checkpoint=checkpoint,
                             validate_observation=True)

    def _coupler(self, brain_state, *, io_scope, checkpoint, validate_observation):
        self._owner()
        from .latent_contract import LatentCoupler
        if io_scope not in {'sensory_motor', 'all_neurons'}:
            raise ValueError('Choose an explicit supported neuron scope')
        path = (ROOT / 'shared_io/checkpoints/neuron_direct' / io_scope / 'current.safetensors'
                if checkpoint is None else Path(checkpoint['path']))
        # Include the catalog and original-model manifest that define the masks
        # and learned weights; a changed file forces all original validations.
        key = (io_scope, file_revision(path),
               file_revision(ROOT / 'artifacts/full_brain/catalog.json'),
               file_revision(ROOT / 'model_training/weights-manifest.json'),
               json.dumps(checkpoint, sort_keys=True) if checkpoint is not None else None)
        if self.entry is not None and self.entry['key'] == key:
            model = self.entry['model']
            if validate_observation:
                model.codec.read(brain_state)  # Synchronous legacy semantics.
                self.brain_checks += 1
            self.hits += 1
            return model
        if self.entry is not None:
            self.invalidations += 1
            self.last_reason = 'scope_checkpoint_catalog_or_model_identity_changed'
        self.entry = None  # Failed replacement cannot fall back to stale weights.
        self.body_embedding_revisions = {}
        started = time.perf_counter()
        model = LatentCoupler(brain_state, io_scope=io_scope, checkpoint=checkpoint)
        if key[1:4] != (file_revision(path),
                file_revision(ROOT / 'artifacts/full_brain/catalog.json'),
                file_revision(ROOT / 'model_training/weights-manifest.json')):
            raise ValueError('Numeric checkpoint identity changed during full verification')
        from .immutable_codec import FrozenCatalogCodec
        model.codec = FrozenCatalogCodec.from_verified(model.codec)
        self.loads += 1
        self.brain_checks += 1
        self.load_seconds += time.perf_counter() - started
        self.entry = {'key': key, 'model': model}
        return model

    def request_heads(self, brain_state, body_packet_descriptor, *,
                      io_scope='sensory_motor', checkpoint=None):
        """Pin immutable versions on owner, return two request-private wrappers.

        Warm owner work is checkpoint fingerprints plus a SHA-checked small
        body JSON. No brain NPZ or feature vector is decoded. The worker must
        export the packet and call condition() before publishing these heads.
        All original per-observation SHA/ID/time checks then run in the worker.
        """
        self._owner()
        from .latent_contract import checked_file
        _, raw = checked_file(body_packet_descriptor)
        packet = json.loads(raw)
        body = self.sensory_sources.body_adapter(packet)
        model = self._coupler(brain_state, io_scope=io_scope, checkpoint=checkpoint,
                              validate_observation=False)
        layout = body.codec.identity['feature_layout_sha256']
        path = ROOT / 'shared_io/checkpoints/body_embedding' / layout / 'current.safetensors'
        revision = file_revision(path)
        if self.body_embedding_revisions.get(layout) != revision:
            # Remove a failed replacement; neither this nor later requests may
            # silently fall back to old weights after an invalid new version.
            model.body_heads.pop(layout, None)
            self.body_embedding_revisions.pop(layout, None)
            model.pin_body_embedding(layout, body.codec.feature_dim)
            if file_revision(path) != revision:
                model.body_heads.pop(layout, None)
                raise ValueError('Body embedding revision changed during verification')
            self.body_embedding_revisions[layout] = revision
            self.body_embedding_loads += 1
        private = copy(body)
        # nn.Module registration dictionaries are mutable too. Its network is
        # a shared frozen inference module; per-request metadata/codec is not.
        for name in ('_parameters', '_buffers', '_modules'):
            setattr(private, name, getattr(body, name).copy())
        private.codec = deepcopy(body.codec)
        private.label_source = deepcopy(body.label_source)
        for name in ('ids', 'indices'):
            value = getattr(body, name).copy();value.setflags(write=False)
            setattr(private, name, value)
        if private.training or any(p.requires_grad for p in private.parameters()):
            raise ValueError('Request sensory head must be frozen')
        self.request_forks += 1
        return model.fork_request(), private

    def status(self):
        entry = self.entry
        return {'schema': 1, 'scope': 'single_lab_owner',
                'policy': 'Session-pinned fully SHA-validated immutable inference versions; '
                          'file revision changes trigger full reload, every observation is freshly validated',
                'trainable_parameters': 0, 'coupler_loads': self.loads, 'coupler_hits': self.hits,
                'invalidations': self.invalidations, 'brain_snapshot_checks': self.brain_checks,
                'coupler_load_seconds': self.load_seconds, 'last_invalidation_reason': self.last_reason,
                'request_private_forks': self.request_forks,
                'body_embedding_loads': self.body_embedding_loads,
                'warm_request_brain_validation': 'Preparation worker; no owner NPZ read',
                'coupler': deepcopy(entry['model'].identity) if entry else None,
                'body_loads': self.sensory_sources.loads, 'body_hits': self.sensory_sources.hits,
                'body_invalidations': self.sensory_sources.invalidations,
                'body_load_seconds': self.sensory_sources.load_seconds,
                'body': deepcopy(self.sensory_sources.last_body)}
