"""Narrow, lossless correction of three demonstrated verifier JSON placements."""
import copy


def normalize(value, ids, first):
    raw = copy.deepcopy(value)
    changes = []
    judgment = raw.get("judgment")
    if not isinstance(judgment, dict):
        return raw, changes
    name = "prior_counterevidence_response"
    if name in raw:
        if name in judgment:
            raise ValueError("ambiguous_counterevidence_locations")
        if isinstance(raw[name], dict):
            judgment[name] = raw.pop(name)
            changes.append("move_top_level_prior_counterevidence_response_into_judgment")
    support = raw.get("support", {})
    conditions = support.get("conditions") if isinstance(support, dict) else None
    first_ids = set(first.get("conditions", {}))
    independent_ids = set(judgment.get("conditions", {}))
    if isinstance(conditions, dict):
        for cid in sorted(set(conditions) - set(ids)):
            if (type(conditions[cid]) is bool and cid not in first_ids and cid not in independent_ids
                    and cid not in {"policy", "assistance", "prior_counterevidence_response"}):
                conditions.pop(cid)
                changes.append("remove_unrequested_support_condition:" + cid)
    issue_support = raw.get("issue_support")
    if isinstance(issue_support, dict):
        expected = {i["id"] for i in first["issues"]}
        independent_issues = {i.get("id") for i in judgment.get("issues", []) if isinstance(i, dict)}
        for iid in sorted(set(issue_support) - expected):
            if iid in independent_issues:
                issue_support.pop(iid)
                changes.append("remove_unrequested_issue_support:" + iid)
    return raw, changes
