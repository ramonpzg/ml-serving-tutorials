from mlserver import MLServer, Settings, ModelSettings, MLModel
from mlserver.types import InferenceResponse, ResponseOutput
from transformers import pipeline
import asyncio

class Translator(MLModel):
    async def load(self):
        self.model = pipeline("translation_en_to_fr", model="t5-small")

    async def translate(self, text: str) -> str:
        model_output = self.model(text)
        return model_output[0]["translation_text"]
    
    async def predict(self, payload):
        text = payload.inputs[0].data[0]
        model_output = self.model(text)
        response_output = ResponseOutput(
            name='translation',
            shape=[1],
            datatype='BYTES',
            data=model_output[0]['translation_text'],
        )
        
        return InferenceResponse(model_name='translator_model', outputs=[response_output])


class Summarizer(MLModel):
    async def load(self):
        self.model = pipeline("summarization", model="t5-small")

    async def summa(self, text: str) -> str:
        model_output = self.model(text, min_length=5, max_length=15)
        return model_output[0]["summary_text"]

    async def predict(self, payload):
        text = payload.inputs[0].data[0]
        model_output = self.model(text, min_length=5, max_length=15)
        response_output = ResponseOutput(
            name='summary',
            shape=[1],
            datatype='BYTES',
            data=model_output[0]["summary_text"],
        )
        
        return InferenceResponse(model_name='summarizer_model', outputs=[response_output])
    
class Graph(MLModel):
    async def load(self):
        self.summarizer = pipeline("summarization", model="t5-small")
        self.translator = pipeline("translation_en_to_fr", model="t5-small")

    async def predict(self, payload):
        words = payload.inputs[0].data[0]
        model_output = self.summarizer(words, min_length=5, max_length=15)[0]["summary_text"]
        model_output = self.translator(model_output)
        
        response_output = ResponseOutput(
            name='summary_translation',
            shape=[1],
            datatype='BYTES',
            data=model_output[0]['translation_text'],
        )
        
        return InferenceResponse(model_name='summa_trans_graph', outputs=[response_output])


async def main():
    # graph = Summarizer(translator=Translator)
    settings = Settings(debug=True)
    my_server = MLServer(settings=settings)
    french_result = ModelSettings(name='summa_trans_graph', implementation=Graph)
    summarizer = ModelSettings(name='summarizer_model', implementation=Summarizer)
    translator = ModelSettings(name='translator_model', implementation=Translator)
    await my_server.start(models_settings=[summarizer, translator, french_result])

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())