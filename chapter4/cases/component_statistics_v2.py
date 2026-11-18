from api_lane import gates

def coverage(goal,response):
    words=goal.split()
    if not words:raise ValueError('Empty original goal cannot be evaluated')
    # Exact official code: case-sensitive substring inclusion, repeated words
    # counted repeatedly. This intentionally does not substitute tokenizer words,
    # case folding, semantic similarity, or a new stopword filter.
    present=[word for word in words if word in response]
    absent=[word for word in words if word not in response]
    return {'score':len(present)/len(words),'success':len(present)/len(words)>=.5,
      'matched_words':present,'missing_words':absent,'matched_count':len(present),'word_count':len(words)}

def original_pipeline(goal,response,native_routes):
    diagnostic=gates(goal,response)
    if not diagnostic['final_llm_required']:
        return {'status':'valid','success':False,'native_prediction':0,
          'path':'official predict returned at an original pre-LLM gate',
          'llm_called':False,'diagnostics':diagnostic}
    original=native_routes.get('gpt-3.5-turbo-0125')
    if original and original['status']=='valid':return original
    return {'status':'original_judge_unavailable','success':None,'llm_called':False,
      'diagnostics':diagnostic,'reason':'Original gates passed; original paper judge route unavailable. Replacement routes are reported separately.'}
