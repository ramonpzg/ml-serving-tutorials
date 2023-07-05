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
            name='new_text',
            shape=[1],
            datatype='BYTES',
            data=model_output[0]['summary_text'],
        )
        
        return InferenceResponse(model_name='test_model', outputs=[response_output])


async def main():
    settings = Settings(debug=True)
    my_server = MLServer(settings=settings)
    implementation = ModelSettings(name='awesome_model', implementation=Summarizer)
    await my_server.start(models_settings=[implementation])

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())