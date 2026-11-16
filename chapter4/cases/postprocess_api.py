import ast
from types import SimpleNamespace
from cases_common import ROOT

def boost_predictor():
    root=ROOT/'references/XLLM'
    ns={};constants=ast.parse((root/'BOOST/utils/constants.py').read_text())
    blacklist=next(x for x in constants.body if isinstance(x,ast.FunctionDef) and x.name=='get_black_list')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[blacklist],type_ignores=[])),'<official BOOST blacklist>','exec'),ns)
    tree=ast.parse((root/'BOOST/Attack_GPTFuzzer/gptfuzzer/llm/llm.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='OpenAILLM')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='predict')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'<official BOOST predict>','exec'),ns)
    tree=ast.parse((root/'BOOST/Attack_ICA/ica.py').read_text())
    prompt=next(x.value for x in ast.walk(tree) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='predict_system_message' for t in x.targets))
    return ns['predict'],prompt

async def boost_check(api,route,question,response,sid):
    predict,prompt_ast=boost_predictor();called=[]
    def fake(text):called.append(text);return ['0']
    native=predict(SimpleNamespace(generate=fake),[response],question)[0]
    if not called:
        return {'native_prediction':native,'path':'official refusal/word-overlap gate; no LLM request necessary',
          'parse_ok':True,'fallback_used':False,'protocol':'official BOOST predict with exact original goal'}
    system=eval(compile(ast.Expression(body=prompt_ast),'<official BOOST judge prompt>','eval'),{'origin_question':question})
    res=await api.call(route,[{'role':'system','content':system},{'role':'user','content':response}],
                        '5B',sid,role='evaluator',max_tokens=512)
    content=res['content'] if res['status']=='ok' else ''
    try:parsed=int(content);parse_ok=parsed in (0,1)
    except ValueError:parse_ok=False
    import contextlib,io
    with contextlib.redirect_stdout(io.StringIO()):
        native=predict(SimpleNamespace(generate=lambda text:[content]),[response],question)[0]
    return {'native_prediction':native,'path':'official gates then original classifier prompt',
      'route':route,'response':res,'parse_ok':parse_ok,'fallback_used':not parse_ok,
      'protocol_difference':'verified relay judge route substituted for original CLI model; parser and gates unchanged'}
