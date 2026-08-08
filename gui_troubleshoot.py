import os
os.environ["TK_SILENCE_DEPRECATION"] = "1"
import sys
import traceback
import time
import tkinter as tk
from tkinter import ttk

class ButtonTextTestApp(tk.Tk):
    """
    Clean, robust GUI test app verifying text generation upon button click.
    Features:
    - StringVar binding for Label display
    - Read-only Text widget using state=tk.NORMAL for updates and state=tk.DISABLED for locking
    - Standard event loop execution (no reentrancy-inducing self.update() calls)
    """
    def __init__(self):
        super().__init__()
        self.title("GUI Button Text Test")
        self.geometry("700x550")
        self.configure(bg="#F1F5F9")

        # Catch callback exceptions cleanly
        self.report_callback_exception = self.handle_callback_exception

        # Bring window to front on launch
        self.lift()
        self.attributes('-topmost', True)
        self.after_idle(self.attributes, '-topmost', False)
        self.focus_force()

        self.click_count = 0

        # Header Banner
        header = tk.Frame(self, bg="#1E3A8A", padx=15, pady=12)
        header.pack(fill=tk.X, side=tk.TOP)

        lbl_title = tk.Label(
            header, text="🧪 GUI Button Text Generation Test",
            font=("Helvetica", 16, "bold"), fg="#FFFFFF", bg="#1E3A8A"
        )
        lbl_title.pack(anchor="w")

        # Controls Panel
        ctrl_frame = tk.Frame(self, bg="#FFFFFF", padx=15, pady=15, relief=tk.RIDGE, bd=1)
        ctrl_frame.pack(fill=tk.X, padx=15, pady=15)

        btn_box = tk.Frame(ctrl_frame, bg="#FFFFFF")
        btn_box.pack(fill=tk.X)

        self.btn_generate = tk.Button(
            btn_box,
            text="▶ GENERATE TEXT",
            command=self.generate_text,
            bg="#2563EB",
            fg="#FFFFFF",
            activebackground="#1D4ED8",
            activeforeground="#FFFFFF",
            highlightbackground="#FFFFFF",
            font=("Helvetica", 12, "bold"),
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.btn_generate.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_clear = tk.Button(
            btn_box,
            text="Clear All",
            command=self.clear_text,
            bg="#DC2626",
            fg="#FFFFFF",
            activebackground="#B91C1C",
            activeforeground="#FFFFFF",
            highlightbackground="#FFFFFF",
            font=("Helvetica", 12, "bold"),
            padx=15,
            pady=10,
            cursor="hand2"
        )
        self.btn_clear.pack(side=tk.LEFT)

        # 1. Label Display bound to StringVar
        self.label_var = tk.StringVar(value="[No text generated yet. Click 'GENERATE TEXT' above!]")

        banner_card = tk.Frame(self, bg="#FEF3C7", padx=15, pady=12, relief=tk.SOLID, bd=1)
        banner_card.pack(fill=tk.X, padx=15, pady=(0, 10))

        lbl_banner_hdr = tk.Label(
            banner_card, text="1. STRINGVAR LABEL DISPLAY:",
            font=("Helvetica", 10, "bold"), fg="#92400E", bg="#FEF3C7"
        )
        lbl_banner_hdr.pack(anchor="w")

        self.lbl_latest_text = tk.Label(
            banner_card,
            textvariable=self.label_var,
            font=("Helvetica", 13, "bold"),
            fg="#1E3A8A",
            bg="#FEF3C7",
            wraplength=650,
            justify="left"
        )
        self.lbl_latest_text.pack(anchor="w", pady=(4, 0))

        # 2. Read-Only Text Area Display
        text_card = tk.Frame(self, bg="#FFFFFF", padx=15, pady=12, relief=tk.RIDGE, bd=1)
        text_card.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        lbl_text_hdr = tk.Label(
            text_card, text="2. READ-ONLY SCROLLED TEXT DISPLAY:",
            font=("Helvetica", 10, "bold"), fg="#0F172A", bg="#FFFFFF"
        )
        lbl_text_hdr.pack(anchor="w", pady=(0, 5))

        text_inner = tk.Frame(text_card, bg="#FFFFFF")
        text_inner.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_inner, orient=tk.VERTICAL)
        
        self.text_area = tk.Text(
            text_inner,
            wrap=tk.WORD,
            bg="#0F172A",
            fg="#00FF66",
            insertbackground="#FFFFFF",
            font=("Courier", 11, "bold"),
            relief=tk.SOLID,
            bd=1,
            yscrollcommand=scrollbar.set,
            state=tk.NORMAL
        )
        scrollbar.config(command=self.text_area.yview)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Initial text insertion & lock to DISABLED (read-only)
        self.text_area.insert(tk.END, "=== GUI READY: CLICK 'GENERATE TEXT' BUTTON ABOVE ===\n")
        self.text_area.see(tk.END)
        self.text_area.config(state=tk.DISABLED)

    def handle_callback_exception(self, exc, val, tb):
        err_msg = "".join(traceback.format_exception(exc, val, tb))
        print(f"ERROR: Exception caught in Tkinter callback:\n{err_msg}", file=sys.stderr, flush=True)

    def generate_text(self):
        try:
            self.click_count += 1
            timestamp = time.strftime("%H:%M:%S")
            generated_msg = f"[{timestamp}] Click #{self.click_count}: Text generated successfully!"

            # Output to Terminal STDOUT
            print(f"TERMINAL LOG: {generated_msg}", flush=True)

            # Update Label via StringVar
            self.label_var.set(generated_msg)

            # Unlock Text widget -> Insert -> Lock to read-only (DISABLED)
            self.text_area.config(state=tk.NORMAL)
            self.text_area.insert(tk.END, generated_msg + "\n")
            self.text_area.see(tk.END)
            self.text_area.config(state=tk.DISABLED)
        except Exception as e:
            print(f"ERROR in generate_text: {e}", file=sys.stderr, flush=True)

    def clear_text(self):
        try:
            self.click_count = 0
            
            # Reset StringVar
            self.label_var.set("[Cleared. Click 'GENERATE TEXT' above!]")
            
            # Clear Text widget (Unlock -> Clear -> Insert -> Lock)
            self.text_area.config(state=tk.NORMAL)
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert(tk.END, "=== CLEARED: CLICK 'GENERATE TEXT' BUTTON ABOVE ===\n")
            self.text_area.see(tk.END)
            self.text_area.config(state=tk.DISABLED)
            
            print("TERMINAL LOG: [Cleared all text]", flush=True)
        except Exception as e:
            print(f"ERROR in clear_text: {e}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    app = ButtonTextTestApp()
    app.mainloop()
