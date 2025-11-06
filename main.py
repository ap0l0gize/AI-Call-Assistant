'''
+-------------------+        +-----------------------+        +------------------+        +------------------------+
|   Step 1: Install |        |  Step 2: Real-Time    |        |  Step 3: Pass    |        |  Step 4: Live Audio    |
|   Python Libraries|        |  Transcription with   |        |  Real-Time       |        |  Stream from ElevenLabs|
+-------------------+        |       AssemblyAI      |        |  Transcript to   |        |                        |
|                   |        +-----------------------+        |      OpenAI      |        +------------------------+
| - assemblyai      |                    |                    +------------------+                    |
| - openai          |                    |                             |                              |
| - elevenlabs      |                    v                             v                              v
| - mpv             |        +-----------------------+        +------------------+        +------------------------+
| - portaudio       |        |                       |        |                  |        |                        |
+-------------------+        |  AssemblyAI performs  |-------->  OpenAI generates|-------->  ElevenLabs streams   |
                             |  real-time speech-to- |        |  response based  |        |  response as live      |
                             |  text transcription   |        |  on transcription|        |  audio to the user     |
                             |                       |        |                  |        |                        |
                             +-----------------------+        +------------------+        +------------------------+
'''

import assemblyai as aai
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingParameters,
    StreamingEvents,
    BeginEvent,
    TurnEvent,
    TerminationEvent,
    StreamingError
)
from elevenlabs import generate, stream
from openai import OpenAI
import json
import sys
from dotenv import load_dotenv
import os

load_dotenv()


class AI_Assistant:
    def __init__(self):
        aai.settings.api_key = os.getenv("AAI_API_KEY")
        self.openai_client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

        self.client = None
        self.user_data = {}

        self.full_transcript = [
            {"role": "system", "content": "You are my call assistant, you take my calls when i'm unavailable. Your goal is to ONLY get name and reason for call from caller."},
        ]


    def start_transcription(self):
        options = StreamingClientOptions(
            api_key=aai.settings.api_key,
            api_host="streaming.assemblyai.com"
        )
        self.client = StreamingClient(options)

        # attach handlers
        self.client.on(StreamingEvents.Begin, self.on_begin)
        self.client.on(StreamingEvents.Turn, self.on_turn)
        self.client.on(StreamingEvents.Termination, self.on_terminated)
        self.client.on(StreamingEvents.Error, self.on_error)

        params = StreamingParameters(
            sample_rate=16000,
            format_turns=True
        )
        self.client.connect(params)
        microphone_stream = aai.extras.MicrophoneStream(sample_rate=16000)
        try:
            self.client.stream(microphone_stream)
        finally:
            self.client.disconnect(terminate=True)


    def on_begin(self, client, event: BeginEvent):
        return

    def on_turn(self, client, event: TurnEvent):
        # final turn received
        text = event.transcript
        is_end = event.end_of_turn
        if is_end:
            self.extract_user_data(text)

    def on_terminated(self, client, event: TerminationEvent):
        return

    def on_error(self, client, error: StreamingError):
        return

    def stop_transcription(self):
        if self.client:
            self.client.disconnect(terminate=True)  # end transcription
            self.client = None


    def extract_user_data(self, transcript):

        self.stop_transcription()

        prompt = f"""
        Extract the user's full name and reason for call from the text below in JSON format, if not given leave empty. 
        But only give plain reason for call. For example, "I'm calling to propose a free loan" → "free loan offer".
        Text: "{transcript}"
        Return only JSON with keys "name" and "reason".
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content": prompt}]
            )
            extracted_json = response.choices[0].message.content

            new_data = json.loads(extracted_json)

            # merge new info with previously captured info
            if not hasattr(self, "user_data"):
                self.user_data = {"name": "", "reason": ""}

            if new_data.get("name"):
                self.user_data["name"] = new_data["name"]
            if new_data.get("reason"):
                self.user_data["reason"] = new_data["reason"]

            # check if either field is missing or empty
            missing_fields = []
            if not self.user_data.get("name"):
                missing_fields.append("name")
            if not self.user_data.get("reason"):
                missing_fields.append("reason")

            if missing_fields:
                # ask again for the missing information
                if "name" in missing_fields and "reason" in missing_fields:
                    self.generate_audio("Could you please tell me your full name and the reason for your call?")
                    self.start_transcription()
                elif "name" in missing_fields:
                    self.generate_audio("Could you please tell me your full name?")
                    self.start_transcription()
                elif "reason" in missing_fields:
                    self.generate_audio("Could you please tell me the reason for your call?")
                    self.start_transcription()
            else:
                # end conversation when all info captured
                print(f"Captured info: {self.user_data}")
                confirmation = (
                    f"Thank you {self.user_data['name']}, "
                    f"Mr Aleksander will reach back to you as soon as possible."
                )
                self.generate_audio(confirmation)
                self.stop_transcription() 
                with open("user_data.json", "w") as f:
                    json.dump(self.user_data, f, indent=2)
                sys.exit(0) # end conversation

        except Exception as e:
            print("Failed to extract user data:", e, "Raw transcript:", transcript)


        

    def generate_audio(self, text):

        self.full_transcript.append({"role":"assistant", "content": text})
        print(f"\nAI Receptionist: {text}")

        audio_stream = generate(
            api_key = self.elevenlabs_api_key,
            text = text,
            voice = "Sarah",
            model="eleven_multilingual_v2",
            stream = True
        )

        stream(audio_stream)

greeting = "Thank you for calling Mr Aleksander. My name is Sandy, please state your name and reason for call."
ai_assistant = AI_Assistant()
ai_assistant.generate_audio(greeting)
ai_assistant.start_transcription()