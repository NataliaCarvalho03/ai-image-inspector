import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import threading
import cv2

from src.engine_cnn import CNNEngine
from src.engine_vlm import VLMEngine

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Image Inspector")
        self.geometry("1100x650")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)

        self.cnn_engine = None
        self.vlm_engine = None

        self.current_image_path = None
        self.current_pil_image  = None
        self.current_cv2_image  = None

        self.setup_left_panel()
        self.setup_right_panel()

        threading.Thread(target=self._load_engines, daemon=True).start()

    def _load_engines(self):
        self.after(0, self._set_status, "Loading CNN engine…")
        self.cnn_engine = CNNEngine()

        self.after(0, self._set_status, "Loading VLM engine… (this may take a moment)")
        self.vlm_engine = VLMEngine()

        self.after(0, self._set_status, "Engines ready.")
        self.after(2000, self._clear_status)

    def _set_status(self, msg: str):
        self.status_label.configure(text=msg)

    def _clear_status(self):
        self.status_label.configure(text="")

    def setup_left_panel(self):
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.left_frame.grid_rowconfigure(0, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(self.left_frame, text="No Image Loaded", justify="center")
        self.image_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.load_btn = ctk.CTkButton(self.left_frame, text="Load Image", command=self.load_image)
        self.load_btn.grid(row=1, column=0, padx=10, pady=(0, 5))

        self.status_label = ctk.CTkLabel(self.left_frame, text="Loading engines…", text_color="gray",
                                         font=ctk.CTkFont(size=11))
        self.status_label.grid(row=2, column=0, padx=10, pady=(0, 10))

    def setup_right_panel(self):
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.tabview = ctk.CTkTabview(self.right_frame)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab1 = self.tabview.add("Object Detection")
        self.tab2 = self.tabview.add("VLM Description")

        self.setup_tab1()
        self.setup_tab2()

    def setup_tab1(self):
        self.detect_btn = ctk.CTkButton(self.tab1, text="Run Object Detection", command=self.run_detection)
        self.detect_btn.pack(pady=15)

        self.results_textbox = ctk.CTkTextbox(self.tab1, state="disabled")
        self.results_textbox.pack(fill="both", expand=True, padx=10, pady=10)

    def setup_tab2(self):
        self.prompt_entry = ctk.CTkEntry(self.tab2, placeholder_text="Enter prompt here…")
        self.prompt_entry.pack(fill="x", padx=10, pady=15)

        self.generate_btn = ctk.CTkButton(self.tab2, text="Generate Description", command=self.run_vlm)
        self.generate_btn.pack(pady=5)

        self.vlm_textbox = ctk.CTkTextbox(self.tab2, state="disabled", wrap="word")
        self.vlm_textbox.pack(fill="both", expand=True, padx=10, pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        self.current_image_path = file_path
        self.current_cv2_image  = cv2.imread(file_path)
        self.current_pil_image  = Image.open(file_path).convert("RGB")

        self.update_idletasks()
        target_w = self.left_frame.winfo_width() - 40
        target_h = self.left_frame.winfo_height() - 100
        if target_w <= 10 or target_h <= 10:
            target_w, target_h = 600, 400

        display_img = self.current_pil_image.copy()
        display_img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=display_img, dark_image=display_img, size=display_img.size)

        self.image_label.configure(image=ctk_img, text="")
        self.image_label.image = ctk_img

    def run_detection(self):
        if self.current_cv2_image is None:
            self.update_textbox(self.results_textbox, "Please load an image first.", False)
            return

        if self.cnn_engine is None:
            self.update_textbox(self.results_textbox, "CNN engine is still loading — please wait.", False)
            return

        self.detect_btn.configure(state="disabled")
        self.update_textbox(self.results_textbox, "Running Object Detection…", False)

        threading.Thread(target=self._detection_thread, daemon=True).start()

    def _detection_thread(self):
        results = self.cnn_engine.detect(self.current_cv2_image)

        if isinstance(results, str):
            text = results
        else:
            text = f"Detected {len(results)} object(s):\n\n"
            for r in results:
                text += f"• Class ID: {r['class_id']} (Confidence: {r['confidence']:.2f})\n"
                text += f"  Bounding Box: {r['box']}\n\n"

        self.after(0, self.update_textbox, self.results_textbox, text, True, self.detect_btn)

    def run_vlm(self):
        if self.current_pil_image is None:
            self.update_textbox(self.vlm_textbox, "Please load an image first.", False)
            return

        if self.vlm_engine is None or not self.vlm_engine.is_loaded():
            self.update_textbox(self.vlm_textbox, "VLM engine is still loading — please wait.", False)
            return

        prompt = self.prompt_entry.get().strip() or "Describe this image in detail."

        self.generate_btn.configure(state="disabled")
        self.update_textbox(self.vlm_textbox, "Generating description… this may take a minute on CPU.", False)

        threading.Thread(target=self._vlm_thread, args=(prompt,), daemon=True).start()

    def _vlm_thread(self, prompt: str):
        result = self.vlm_engine.generate_description(self.current_pil_image, prompt)
        self.after(0, self.update_textbox, self.vlm_textbox, result, True, self.generate_btn)

    def update_textbox(self, textbox, text, enable_btn=False, btn_widget=None):
        textbox.configure(state="normal")
        textbox.delete("1.0", "end")
        textbox.insert("end", text)
        textbox.configure(state="disabled")

        if enable_btn and btn_widget:
            btn_widget.configure(state="normal")
