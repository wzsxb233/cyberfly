"""Actual filesystem checks; fixtures are bytes, never model observations."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vllm_integration import checked_artifacts as reader


class CheckedArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='cyberfly-checked-artifacts-')
        self.addCleanup(temporary.cleanup)
        self.addCleanup(reader.close_cached_roots)
        self.base=Path(temporary.name);self.root=self.base/'trusted';self.root.mkdir()
        self.path=self.root/'data.bin';self.path.write_bytes(b'first data')

    def descriptor(self,path=None,raw=None):
        path=path or self.path
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes() if raw is None else raw).hexdigest()}

    def test_same_size_replacement_is_always_rehashed(self):
        descriptor=self.descriptor()
        self.assertEqual(reader.read_checked_artifact(descriptor,root=self.root)[1],b'first data')
        self.path.write_bytes(b'other data')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):
            reader.read_checked_artifact(descriptor,root=self.root)
        self.assertEqual(reader.read_checked_artifact(self.descriptor(),root=self.root)[1],b'other data')

    def test_outside_and_dotdot_escape_rejected(self):
        outside=self.base/'outside';outside.write_bytes(b'outside')
        for path in (outside,self.root/'..'/'outside'):
            with self.assertRaisesRegex(ValueError,'outside|escapes'):
                reader.read_checked_artifact(self.descriptor(path),root=self.root)

    def test_openat2_rejects_file_and_directory_symlinks(self):
        reader.read_checked_artifact(self.descriptor(),root=self.root)
        if reader.read_backend()!='linux_openat2':self.skipTest('Kernel lacks openat2')
        (self.root/'link').symlink_to(self.path)
        (self.root/'parent-link').symlink_to(self.root,target_is_directory=True)
        for path in (self.root/'link',self.root/'parent-link'/'data.bin'):
            with self.assertRaisesRegex(ValueError,'symbolic link'):
                reader.read_checked_artifact(self.descriptor(path),root=self.root)

    def test_limit_and_nonregular_files_rejected(self):
        with self.assertRaisesRegex(ValueError,'oversized'):
            reader.read_checked_artifact(self.descriptor(),root=self.root,limit=3)
        for path in (self.root,self.root/'pipe'):
            if path!=self.root:os.mkfifo(path)
            with self.assertRaisesRegex(ValueError,'oversized'):
                reader.read_checked_artifact({'path':str(path),'sha256':'0'*64},root=self.root)

    def test_cached_root_replacement_opens_new_identity(self):
        descriptor=self.descriptor();reader.read_checked_artifact(descriptor,root=self.root)
        self.root.rename(self.base/'old');self.root.mkdir();self.path.write_bytes(b'other data')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):
            reader.read_checked_artifact(descriptor,root=self.root)
        self.assertEqual(reader.read_checked_artifact(self.descriptor(),root=self.root)[1],b'other data')

    def test_bounded_root_cache_closes_evicted_descriptors(self):
        for index in range(12):
            root=self.base/str(index);root.mkdir();path=root/'file';path.write_bytes(b'input')
            reader.read_checked_artifact(self.descriptor(path),root=root)
        self.assertLessEqual(len(reader._roots),8)

    def test_old_kernel_fallback_still_checks_root_and_every_sha(self):
        with patch.object(reader,'_openat2_available',False):
            descriptor=self.descriptor()
            self.assertEqual(reader.read_checked_artifact(descriptor,root=self.root)[1],b'first data')
            self.path.write_bytes(b'other data')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):
                reader.read_checked_artifact(descriptor,root=self.root)
            with self.assertRaisesRegex(ValueError,'outside'):
                reader.read_checked_artifact(descriptor,root=self.base/'absent')


if __name__=='__main__':unittest.main()
