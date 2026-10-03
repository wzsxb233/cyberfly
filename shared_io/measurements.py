"""Compact measured facts; full tensors and provenance remain in the event."""
def compact_measurements(brain_result,body_feedback=None,body_state=None):
    def pick(value,keys):
        value=value or {}
        return {key:value[key] for key in keys if key in value}
    def rounded(value):
        if isinstance(value,float):return round(value,6)
        if isinstance(value,list):return [rounded(v) for v in value]
        if isinstance(value,dict):return {k:rounded(v) for k,v in value.items()}
        return value
    return rounded({
        'brain':pick(brain_result.get('neural'),('duration_ms','simulation_ms','spikes_this_step')),
        'body':pick(body_feedback or body_state,('interval_s','displacement_mm','horizontal_displacement_mm','contacts_before','contacts','contact_count_change')),
        'feedback':pick(brain_result.get('stimulus'),('delivered_ms_this_step','pending_ms')),
        'learning':pick(brain_result.get('learning'),('enabled','changed_edges')),
        'instruction':'Only describe measured values. Negative contact change means fewer contacts. Two positions do not prove oscillation or intent.'})
