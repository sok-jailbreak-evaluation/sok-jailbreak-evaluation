import io,contextlib
from postprocess_api import boost_predictor

def gates(goal,response):
    predict,_=boost_predictor();called=[]
    with contextlib.redirect_stdout(io.StringIO()):
        verdict=predict(type('Gate',(),{'generate':lambda self,text:called.append(text) or ['0']})(),[response],goal)[0]
    split=goal.split()
    return {'final_llm_required':bool(called),'word_overlap':sum(w in response for w in split)/len(split),
      'early_gate_prediction':None if called else verdict}
