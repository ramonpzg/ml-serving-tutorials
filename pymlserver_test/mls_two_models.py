from mlserver import MLServer, Settings, ModelSettings, MLModel
from mlserver.types import InferenceResponse, ResponseOutput
from transformers import pipeline
import asyncio

class Summarizer(MLModel):
    async def load(self):
        self.model = pipeline("summarization", model="t5-small")

    async def predict(self, payload):
        text = payload.inputs[0].data[0]
        model_output = self.model(text, min_length=5, max_length=15)
        response_output = ResponseOutput(
            name='summary',
            shape=[1],
            datatype='BYTES',
            data=model_output[0]['summary_text'],
        )
        
        return InferenceResponse(model_name='summarizer_model', outputs=[response_output])

class Translator(MLModel):
    async def load(self):
        self.translator = pipeline("translation_en_to_fr", model="t5-small")

    async def predict(self, payload):
        text = payload.inputs[0].data[0]
        model_output = self.translator(text)
        response_output = ResponseOutput(
            name='translation',
            shape=[1],
            datatype='BYTES',
            data=model_output[0]['translation_text'],
        )
        
        return InferenceResponse(model_name='translator_model', outputs=[response_output])


async def main():
    settings = Settings(debug=True)
    my_server = MLServer(settings=settings)
    summarizer = ModelSettings(name='summarizer_model', implementation=Summarizer)
    translator = ModelSettings(name='translator_model', implementation=Translator)
    await my_server.start(models_settings=[summarizer, translator])

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())