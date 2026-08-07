# Imports

import re
import torch
import torch.nn as nn
import gradio as gr

DEVICE = torch.device("cpu")


# Load checkpoint

CHECKPOINT_PATH = "lstm_checkpoint.pt"

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

config = checkpoint["model_config"]

vocab = checkpoint["vocab"]

MAX_LENGTH = config["max_length"]


# Preprocessing

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return clean_text(text).split()


def text_to_ids(text):
    tokens = tokenize(text)[:MAX_LENGTH]

    ids = [vocab.get(t, 1) for t in tokens]

    ids += [0] * (MAX_LENGTH - len(ids))

    return ids


# LSTM


class SimpleLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        dropout,
        embedding_matrix=None,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0,
        )

        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(
                    embedding_matrix,
                    dtype=torch.float32,
                )
            )

        self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, input_ids):

        x = self.dropout(
            self.embedding(input_ids)
        )

        out, _ = self.lstm(x)

        pooled = out.mean(dim=1)

        logits = self.fc(
            self.dropout(pooled)
        )

        return logits


# Load model

model = SimpleLSTM(
    vocab_size=config["vocab_size"],
    embed_dim=config["embed_dim"],
    hidden_dim=config["hidden_dim"],
    num_layers=config["num_layers"],
    dropout=config["dropout"],
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)

model.eval()


# Score a single option

def score_option(question, option):
    text = clean_text(question) + " " + clean_text(option)

    ids = text_to_ids(text)

    input_ids = torch.tensor(
        [ids],
        dtype=torch.long,
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids)
        prob = torch.softmax(logits, dim=-1)[0, 1].item()

    return prob

# Predict all five options

def predict(question, option_a, option_b, option_c, option_d, option_e):

    options = {
        "A": option_a,
        "B": option_b,
        "C": option_c,
        "D": option_d,
        "E": option_e,
    }

    scores = {}

    for letter, text in options.items():
        scores[letter] = score_option(question, text)

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    result = ""

    for rank, (letter, score) in enumerate(ranked, start=1):
        result += f"{rank}. Option {letter} : {score:.6f}\n"

    top3 = " ".join([x[0] for x in ranked[:3]])

    result += "\nTop-3 Prediction : " + top3

    return result


css = """
/* Make the app wider */
.gradio-container {
    max-width: 95% !important;
    margin: auto;
}

/* Hide Gradio footer */
footer {
    display: none;
}

/* Main title */
h1 {
    text-align: center;
    color: #1565C0;
    font-size: 36px !important;
    font-weight: bold !important;
    margin-bottom: 10px;
}

/* Description */
.prose {
    font-size: 18px !important;
    text-align: center;
}

/* Labels: Question, Options, Prediction */
label {
    font-size: 18px !important;
    font-weight: bold !important;
    color: #0D47A1 !important;
}

/* Text areas */
textarea {
    border-radius: 12px !important;
    border: 2px solid #90CAF9 !important;
    font-size: 16px !important;
    line-height: 1.6 !important;
}

/* Buttons */
button {
    font-size: 18px !important;
    font-weight: bold !important;
    border-radius: 10px !important;
}

/* Output box */
.output textarea {
    background-color: #F5F9FF !important;
    font-weight: bold;
}
"""


# Examples

examples=[
    [
        "Which of the following is correct? What is the SI base unit of time and how is it defined? carefully.",
        "The SI base unit of time is the week, which is defined by measuring the electronic transition frequency of caesium atoms.",
        "The SI base unit of time is the second, which is defined by measuring the electronic transition frequency of caesium atoms.",
        "The SI base unit of time is the hour, which is defined by measuring the electronic transition frequency of caesium atoms.",
        "The SI base unit of time is the day, which is defined by measuring the electronic transition frequency of caesium atoms.",
        "The SI base unit of time is the minute, which is defined by measuring the electronic transition frequency of caesium atoms.",
    ],
    ["Which of the following is correct? Which hand should be used to apply the right-hand rule when tightening or loosening nuts, screws, bolts, bottle caps, and jar lids? from the following choices.",
     "One's dominant hand",
     "The right hand",
     "Both hands",
     "The left hand",
     "Either hand",],
    [
        "Select the most accurate option: What is the relationship between mass, force, and acceleration, according to Sir Isaac Newton's laws of motion? among the listed options.",
        "Mass is a property that determines the weight of an object. According to Newton's laws of motion and the formula F = ma, an object with a mass of one kilogram accelerates at one meter per second per second when acted upon by a force of one newton.",
        "Mass is an inertial property that determines an object's tendency to remain at constant velocity unless acted upon by an outside force. According to Newton's laws of motion and the formula F = ma, an object with a mass of one kilogram accelerates at ten meters per second per second when acted upon by a force of one newton.",
        "Mass is an inertial property that determines an object's tendency to remain at constant velocity unless acted upon by an outside force. According to Newton's laws of motion and the formula F = ma, an object with a mass of one kilogram accelerates at ten meters per second per second when acted upon by a force of ten newtons.",
        "Mass is an inertial property that determines an object's tendency to remain at constant velocity unless acted upon by an outside force. According to Newton's laws of motion and the formula F = ma, an object with a mass of one kilogram accelerates at one meter per second per second when acted upon by a force of one newton.",
        "Mass is a property that determines the size of an object. According to Newton's laws of motion and the formula F = ma, an object with a mass of one kilogram accelerates at one meter per second per second when acted upon by a force of ten newtons.",
    ],
]


theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
)

# Gradio Interface


demo = gr.Interface(
    css=css,
    theme=theme,
    fn=predict,
    inputs=[
        gr.Textbox(label="Question"),
        gr.Textbox(label="Option A"),
        gr.Textbox(label="Option B"),
        gr.Textbox(label="Option C"),
        gr.Textbox(label="Option D"),
        gr.Textbox(label="Option E"),
    ],
    outputs=gr.Textbox(
        label="Prediction",
        lines=10,
    ),
    examples=examples,
    title="Physics MCQ Solver",

    description="Enter a physics multiple-choice question and its five options. The system ranks the options based on their predicted likelihood of being correct.",
)


# Launch

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
