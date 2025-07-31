from dispatcher import Dispatcher
from llm_main import LLMRouter

# Dummy classifier for demonstration
class Classifier:
    def classify(self, raw_query):
        # Simple keyword-based classifier
        if 'snapshot' in raw_query or 'Asaoka' in raw_query:
            return 'Asaoka_data', 'Asaoka_data'
        elif 'report' in raw_query or 'doc' in raw_query:
            return 'reporter_Asaoka', 'reporter_Asaoka'
        elif 'plot' in raw_query or 'graph' in raw_query:
            return 'plot_combi_S', 'plot_combi_S'
        else:
            return None, None

if __name__ == "__main__":
    router = LLMRouter(max_turns=3)
    classifier = Classifier()
    disp = Dispatcher(classifier, router)

    while True:
        raw = input(">> ")
        func_name, func = disp.classify(raw)
        if func_name:
            params = disp.gather_params(raw, func_name)
            out = disp.run_function(func_name, params) if params else None
        else:
            params, out = {}, None
        disp.build_and_send(raw, func_name, params, out)
