import functools
import threading
import os

import google.generativeai as client
# Set API key cho google-generativeai
client.configure(api_key=os.environ.get('GEMINI_API_KEY')) 

import ida_kernwin

from gepetto.models.base import LanguageModel
import gepetto.models.model_manager
import gepetto.config

GEMINI_15_FLASH = "gemini-1.5-flash"
GEMINI_15_FLASH_002 = "gemini-1.5-flash-002"
GEMINI_MODELS = [GEMINI_15_FLASH, GEMINI_15_FLASH_002]

# Khởi tạo Gemini client
def create_client():
    """
    Initialize and return the Gemini client.
    """
    return client

class Gemini(LanguageModel):
    """
    Gemini language model class.
    """

    def __init__(self, model):
        """
        Initialize the Gemini model with the specified model name.

        Args:
            model (str): The name of the Gemini model to use.
        """
        self.model = model
        # Set API key for google-generativeai
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.client = create_client()

    @staticmethod
    def get_menu_name() -> str:
        """
        Get the menu name for the Gemini model.

        Returns:
            str: The menu name for the Gemini model.
        """
        return "Gemini"

    @staticmethod
    def supported_models():
        global GEMINI_MODELS
        if GEMINI_MODELS is None:
            try:
                gemini_client = create_client()
                # Lấy danh sách các model Gemini
                models = [m.name for m in gemini_client.list_models()]
                GEMINI_MODELS = [m for m in models if 'gemini' in m]
            except Exception as e:
                print(f"Lỗi khi lấy danh sách models: {e}")
                GEMINI_MODELS = []
            print(f"Models: {GEMINI_MODELS}")
        return GEMINI_MODELS

    def __str__(self):
        return self.model

    def query_model_async(self, query, cb, additional_model_options=None):
        if additional_model_options is None:
            additional_model_options = {}
        t = threading.Thread(target=self.query_model, args=[query, cb, additional_model_options])
        t.start()

    def query_model(self, query, cb, additional_model_options=None):
        try:
            conversation = query

            model = self.client.GenerativeModel(model_name=self.model)
            # Gọi API Gemini
            #print(f"\nConversation: {conversation}\n")
            response = model.generate_content(contents=conversation)
            
            # fix response
            resp_text = response.text.replace('```json\n', '').replace('```', '')

            #print(f"\nResponse: {resp_text}\n")
            
            ida_kernwin.execute_sync(functools.partial(cb, response=resp_text),
                                     ida_kernwin.MFF_WRITE)
        except Exception as e:
            print(f"Error while calling Gemini API: {e}")

gepetto.models.model_manager.register_model(Gemini)