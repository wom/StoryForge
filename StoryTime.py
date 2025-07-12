import os
from textual.app import App, ComposeResult
from textual.widgets import Input, Button, Static, Log, Label, LoadingIndicator
from textual.containers import Vertical, Horizontal
from textual.widget import Widget
from textual.reactive import reactive
from textual.events import Key
import asyncio
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64

# Set up Gemini client
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")

client = genai.Client(api_key=GEMINI_API_KEY)

class StoryApp(App):
    CSS_PATH = None
    BINDINGS = [ ("q", "quit", "Quit") ]

    async def on_mount(self) -> None:
        # No spinner to hide on startup anymore
        pass

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("Enter a story prompt:"),
            Input(placeholder="Type your story prompt here...", id="prompt_input"),
            Button("Generate Story & Image", id="generate_btn"),
            Log(id="output_log", highlight=True, max_lines=100),
            # Confirmation dialog, hidden by default
            Static("", id="confirm_dialog", classes="hidden"),

        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "generate_btn":
            prompt_input = self.query_one("#prompt_input", Input)
            output_log = self.query_one("#output_log", Log)
            prompt = prompt_input.value.strip()
            if not prompt:
                output_log.write("[red]Please enter a story prompt.[/red]")
                return
            # Show confirmation dialog
            confirm_dialog = self.query_one("#confirm_dialog", Static)
            confirm_dialog.update(
                "[bold]Are you sure?[/bold]\n\nThis will generate a story and an image based on your prompt, and save the image to disk.\n\nContinue?\n\n"
            )
            confirm_dialog.remove_class("hidden")
            confirm_dialog.mount(Horizontal(
                Button("Yes", id="confirm_yes"),
                Button("No", id="confirm_no")
            ))
            self.set_focus(confirm_dialog)
            self._pending_prompt = prompt
            return
        elif event.button.id == "confirm_yes":
            await self._do_generation()
            confirm_dialog = self.query_one("#confirm_dialog", Static)
            confirm_dialog.update("")
            confirm_dialog.add_class("hidden")
            # Remove all children (buttons) from dialog
            for child in list(confirm_dialog.children):
                child.remove()
        elif event.button.id == "confirm_no":
            confirm_dialog = self.query_one("#confirm_dialog", Static)
            confirm_dialog.update("")
            confirm_dialog.add_class("hidden")
            # Remove all children (buttons) from dialog
            for child in list(confirm_dialog.children):
                child.remove()
            # Remove spinner if present
            try:
                spinner = self.query("#working_spinner").first()
                if spinner is not None:
                    spinner.remove()
            except Exception:
                pass
            self._pending_prompt = None

    async def _do_generation(self):
        prompt = getattr(self, "_pending_prompt", None)
        if not prompt:
            return
        output_log = self.query_one("#output_log", Log)
        spinner = LoadingIndicator(id="working_spinner")
        self.mount(spinner)
        try:
            output_log.clear()
            output_log.write(f"[bold]Generating story for:[/bold] {prompt}")
            await asyncio.sleep(0.1)  # Let spinner show
            # Generate story
            story = await asyncio.to_thread(self.generate_story, prompt)
            output_log.write(f"[green]Story:[/green]\n{story}")
            output_log.write("[bold]Generating image...[/bold]")
            image, image_bytes = await asyncio.to_thread(self.generate_image, prompt)
            if image is None:
                output_log.write("[red]Failed to generate image.[/red]")
                return
            output_log.write("[bold]Generating image name...[/bold]")
            image_name = await asyncio.to_thread(self.generate_image_name, prompt, story)
            if not image_name:
                image_name = "story_image.png"
            else:
                image_name = image_name.replace(" ", "_").replace("/", "-") + ".png"
            with open(image_name, "wb") as f:
                f.write(image_bytes)
            output_log.write(f"[green]Image saved as:[/green] {image_name}")
        finally:
            spinner.remove()
            self._pending_prompt = None

    def generate_story(self, prompt: str) -> str:
        # This runs in a thread, so it's blocking
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Write a short story based on this prompt: {prompt}"
        )
        try:
            return response.candidates[0].content.parts[0].text.strip()
        except Exception:
            return "[Error generating story]"

    def generate_image(self, prompt: str):
        # This runs in a thread, so it's blocking
        contents = f"Create a detailed, beautiful illustration for this story: {prompt}"
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                try:
                    image = Image.open(BytesIO(part.inline_data.data))
                    image_bytes = part.inline_data.data
                    return image, image_bytes
                except Exception:
                    continue
        return None, None

    def generate_image_name(self, prompt: str, story: str) -> str:
        # This runs in a thread, so it's blocking
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Given this story: {story}\nSuggest a short, creative, and descriptive filename for an image illustrating it (no spaces, no special characters, just letters, numbers, and underscores)."
        )
        try:
            name = response.candidates[0].content.parts[0].text.strip()
            # Remove any file extension if present
            name = name.split(".")[0]
            return name
        except Exception:
            return "story_image"

if __name__ == "__main__":
    StoryApp().run()

# CSS for hiding/showing dialog/spinner (if using textual CSS, otherwise can be handled in code)
