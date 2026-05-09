from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

BASE = "/home/likith/Desktop/College/6th Sem/Gen Ai"
OUTPUT = BASE + "/GenAI_All_Labs.pdf"

# ── (file_path, lab_number, title, description, {line_no: comment}) ──
LABS = [

(BASE+"/lab1/lab1.py", 1,
 "Word Embeddings – Analogy & Similarity (GloVe)",
 "Loads a pre-trained 50-dim GloVe model and performs word vector arithmetic analogies.",
 {
  1:  "Import gensim downloader to fetch pre-trained embedding models",
  4:  "Print a loading message – model download can take a while",
  5:  "Load the 50-dim GloVe model trained on Wikipedia + Gigaword corpus",
  8:  "Define function ewr() that runs all embedding analogy experiments",
  9:  "Vector math: king - man + woman → nearest word should be 'queen'",
  10: "Print the top-1 analogy result word",
  11: "Print the cosine similarity score (0–1) of the result",
  13: "Geography analogy: paris - france + italy → expected answer 'rome'",
  14: "Print the capital-of-Italy analogy result",
  15: "Print its similarity score",
  17: "Find the 5 words closest to 'programming' in the embedding space",
  18: "Print header for the nearest-neighbour results",
  19: "Iterate over each (word, score) pair in the result",
  20: "Print each similar word and its cosine similarity value",
  23: "Call ewr() to execute all the analogy and similarity experiments",
 }),

(BASE+"/lab2/lab2.py", 2,
 "Word Embedding Visualisation with PCA",
 "Fetches GloVe vectors for 10 tech words, reduces them to 2-D with PCA, and plots them.",
 {
  1:  "Import matplotlib for scatter-plot visualisation",
  2:  "Import PCA to reduce 50-D vectors to 2-D",
  3:  "Import gensim downloader",
  7:  "Function rd(): fits PCA on embeddings and returns 2-D coordinates",
  8:  "Create PCA object targeting 2 principal components",
  9:  "Fit PCA on the embedding matrix and project to 2-D",
  10: "Return the 2-D reduced array",
  14: "Function visualize(): scatter-plots each word at its 2-D position",
  15: "Create a 10×6 inch figure canvas",
  16: "Iterate over words with their index for annotation",
  17: "Unpack the 2-D (x, y) coordinate for the current word",
  18: "Draw a blue dot at the word's position in 2-D space",
  19: "Label the dot with the word name, offset slightly so it doesn't overlap",
  20: "Render and display the completed scatter plot",
  23: "Function gsm(): prints top-5 nearest neighbours of a given word",
  24: "Retrieve 5 closest words in embedding space",
  25: "Iterate over (word, score) results",
  26: "Print each neighbour and its similarity",
  29: "Load the 50-dim GloVe model",
  31: "Define list of 10 technology-domain words to analyse",
  34: "Fetch the 50-D embedding vector for each word in the list",
  35: "Reduce all embeddings from 50-D to 2-D using PCA",
  36: "Plot the 2-D word positions",
  37: "Print 5 nearest neighbours of 'hardware'",
 }),

(BASE+"/lab3/lab3.py", 3,
 "Custom Word2Vec on Medical Corpus",
 "Trains a Word2Vec model from scratch on 10 medical sentences, then visualises embeddings with PCA.",
 {
  1:  "Import Word2Vec model class from gensim",
  2:  "Import PCA for 2-D projection",
  3:  "Import matplotlib for plotting",
  6:  "Define 10 medical sentences as the raw training corpus",
  20: "Tokenise corpus: lowercase each sentence and split into word list",
  21: "Train Word2Vec: 5-D vectors, window=2, min_count=1, 1000 epochs",
  22: "Ask user for a query word and convert to lowercase",
  25: "Check if the word exists in the trained vocabulary",
  26: "Find the 5 most similar words in the trained embedding space",
  28: "Iterate with 1-based rank index",
  29: "Print rank, similar word, and similarity score",
  31: "Inform user if queried word is not in the vocabulary",
  34: "Get all vocabulary words as an ordered list",
  35: "Retrieve the full embedding matrix for all vocabulary words",
  36: "Create PCA object for 2 output dimensions",
  37: "Project all word vectors to 2-D",
  38: "Create a wide 20×8 inch figure",
  39: "Scatter-plot every word at its 2-D PCA coordinates",
  41: "Iterate over words for text annotation",
  42: "Label each scatter point with the corresponding word",
  45: "Set chart title",
  46: "Label x-axis as first principal component",
  47: "Label y-axis as second principal component",
  48: "Show grid lines for easier reading",
  49: "Render and display the plot",
 }),

(BASE+"/lab4/lab4.py", 4,
 "Prompt Enrichment with GloVe + GPT-2 Text Generation",
 "Replaces a target keyword in a prompt with its closest GloVe neighbour, then compares GPT-2 outputs.",
 {
  1:  "Import gensim downloader for pre-trained word vectors",
  2:  "Import HuggingFace pipeline helper for text generation",
  3:  "Import nltk for tokenisation and string for punctuation constants",
  5:  "Import NLTK's word-level tokeniser",
  6:  "Download Punkt tokeniser data if not already installed",
  7:  "Print loading notice for the GloVe model",
  8:  "Load 100-dim GloVe embeddings (richer than 50-dim)",
  10: "Define function to replace a keyword with its nearest GloVe word",
  11: "Tokenise the prompt string into individual word tokens",
  12: "Initialise list to hold the modified token sequence",
  13: "Loop over each token in the prompt",
  14: "Lowercase and strip punctuation for clean keyword matching",
  15: "Check if this token matches the target keyword",
  17: "Fetch the single closest GloVe word for the keyword",
  18: "Check a similar word was actually found",
  19: "Take the top-ranked similar word as the replacement",
  20: "Log the substitution being made",
  21: "Add the replacement word to the output token list",
  22: "Skip appending the original keyword word",
  23: "Handle case where keyword is not in the GloVe vocabulary",
  25: "Fall back to keeping the original word",
  27: "Re-join the modified tokens back into a prompt string",
  28: "Print the enriched prompt for inspection",
  29: "Return the enriched prompt string",
  32: "Print a notice before loading GPT-2",
  33: "Load GPT-2 text-generation pipeline from HuggingFace",
  35: "Define function to generate text from any prompt using GPT-2",
  37: "Call GPT-2 with max 100 tokens and request 1 output sequence",
  38: "Return the generated text string",
  39: "Catch any inference errors gracefully",
  40: "Print the error message and return None",
  43: "Define the base prompt to test",
  44: "Print the original prompt",
  45: "Set 'disaster' as the keyword to replace",
  46: "Create the enriched prompt by substituting the keyword",
  47: "Print label before generating original response",
  48: "Generate GPT-2 text for the original prompt",
  49: "Print the original generated text",
  51: "Print label before generating enriched response",
  52: "Generate GPT-2 text for the enriched prompt",
  53: "Print the enriched generated text",
  55: "Print section header for comparison metrics",
  56: "Compare the character length of both responses",
  57: "Compare length of enriched response",
  58: "Count sentence-ending periods as a detail-richness proxy",
  59: "Count periods in enriched response",
 }),

(BASE+"/lab5/lab5.py", 5,
 "Story Generation using GloVe Word Similarities",
 "Uses a seed word's 50 nearest GloVe neighbours to fill slots in a short story template.",
 {
  1:  "Import random module for selecting words randomly from a list",
  2:  "Import gensim downloader",
  5:  "Print loading message",
  6:  "Load the 200-dim GloVe model – larger vocab and richer semantics",
  8:  "Define function to retrieve the top-N similar words for a seed",
  11: "Get top-N nearest neighbours, discarding similarity scores",
  12: "Continuation of the most_similar call (line wrapped)",
  13: "Return the list of similar words",
  14: "Handle case where seed word is absent from the vocabulary",
  15: "Return empty list so caller can handle gracefully",
  17: "Define function to build a paragraph-length story",
  18: "Docstring describing the function's purpose",
  19: "Fetch similar words to use as story vocabulary",
  21: "Return a friendly error message if no similar words found",
  22: "End of fallback message string",
  24: "Begin constructing the story paragraph using an f-string template",
  25: "Fill first story slot with a random similar word",
  26: "Fill second and third story slots with random similar words",
  27: "Fill fourth story slot with a random similar word",
  29: "Return the completed story paragraph",
  32: "Prompt user to enter a seed word and normalise to lowercase",
  33: "Print a header before the generated story",
  34: "Generate and print the story paragraph",
 }),

(BASE+"/lab6/lab6.py", 6,
 "Sentiment Analysis with HuggingFace Transformers",
 "Interactive loop that classifies the sentiment of user-entered sentences using a pre-trained transformer.",
 {
  1:  "Import HuggingFace pipeline for easy model inference",
  3:  "Comment: load the sentiment-analysis pipeline",
  4:  "Load default sentiment model (DistilBERT fine-tuned on SST-2 dataset)",
  7:  "Define function that runs the model and returns a formatted result string",
  8:  "Pass the input text through the sentiment model",
  9:  "Extract predicted label: 'POSITIVE' or 'NEGATIVE'",
  10: "Extract confidence score as a float between 0 and 1",
  11: "Return a formatted string showing label and score to 2 decimal places",
  14: "Start an infinite loop to keep accepting user input",
  15: "Read a sentence from the user and remove leading/trailing whitespace",
  16: "Break out of the loop when user types 'exit'",
  17: "Otherwise run sentiment analysis and print the result",
 }),

(BASE+"/lab7/lab7.py", 7,
 "Text Summarisation with BART",
 "Loads facebook/bart-large-cnn and summarises a long air-pollution article using beam search.",
 {
  1:  "Import AutoModelForSeq2SeqLM and AutoTokenizer from HuggingFace transformers",
  3:  "Specify the BART model fine-tuned for summarisation on CNN/DailyMail dataset",
  4:  "Download and initialise the BART tokeniser from HuggingFace Hub",
  5:  "Download and load the BART seq2seq model weights into memory",
  7:  "Begin defining the long input article as a multi-line string",
  31: "Comment: tokenize the input text",
  32: "Encode article to PyTorch tensor, truncate to BART's 512-token limit",
  34: "Comment: generate the summary",
  35: "Run beam-search generation: 4 beams, output capped between 25–50 tokens",
  36: "length_penalty>1 encourages longer summaries; early_stopping halts when all beams finish",
  38: "Decode the generated token IDs back to a readable string, removing special tokens",
  40: "Print the final summarised text to stdout",
 }),

(BASE+"/lab9/lab9.py", 9,
 "Wikipedia Institution Scraper with Pydantic",
 "Fetches a Wikipedia page, extracts founder/year/branches/employees via regex, and validates with Pydantic.",
 {
  1:  "Import re module for regular expression pattern matching",
  2:  "Import wikipedia-api for programmatic Wikipedia page access",
  3:  "Import BaseModel and Field from Pydantic for data validation",
  4:  "Import List and Optional type hints",
  8:  "Define Pydantic data model representing one institution's details",
  9:  "Required string field for the institution's name",
  10: "Optional founder field; defaults to None if not found",
  11: "Optional founding year as string; defaults to None",
  12: "Optional list of branch locations; defaults to empty list",
  13: "Optional integer for employee count; defaults to None",
  14: "Optional 500-character summary snippet from Wikipedia",
  18: "Main extraction function; returns a validated InstitutionDetails object",
  19: "Comment: setup Wikipedia API with a proper User-Agent",
  20: "Set a descriptive User-Agent string as required by Wikipedia's API policy",
  21: "Initialise Wikipedia API client for English language pages",
  22: "Fetch the Wikipedia page object for the given institution name",
  24: "Raise a ValueError if the page does not exist on Wikipedia",
  27: "Store the full plain-text content of the Wikipedia article",
  30: "Comment: helper function to extract info using regex",
  31: "Run a case-insensitive regex search on the article text",
  32: "Extract the first capture group from the match object",
  34: "If is_list=True, split the match on commas to produce a list",
  36: "Return the matched string (or None if no match)",
  37: "Return empty list or None depending on is_list flag",
  40: "Comment: extraction logic",
  41: "Regex pattern to capture the name after 'founded/established/started by'",
  43: "Run the founder regex search on the full article text",
  44: "Extract founder name or default to 'Unknown'",
  46: "Regex to capture a 4-digit year after founding-related keywords",
  48: "Run the year regex search on the article text",
  50: "Extract founding year string or default to 'Unknown'",
  52: "Extract branch locations as a comma-split list",
  54: "Extract raw employee count string (may contain commas)",
  55: "Initialise employee count as None",
  56: "Only proceed if the regex found an employee number",
  57: "Remove commas and convert to integer",
  58: "Handle case where the extracted string cannot be parsed as int",
  62: "Construct and return the validated Pydantic model with all extracted fields",
  63: "Use the Wikipedia page's official title as the name",
  64: "Pass extracted founder string",
  65: "Pass extracted founding year",
  66: "Pass extracted branches list",
  67: "Pass parsed employee count integer",
  68: "Attach first 500 chars of the Wikipedia summary with ellipsis",
  72: "Comment: Execution block",
  73: "Wrap in try/except to handle pages that don't exist or parse errors",
  74: "Prompt user to enter an institution name",
  75: "Call the main scraper function",
  76: "Pretty-print the Pydantic model as indented JSON",
  77: "Catch any exception and print its message",
  78: "Print the error to the user",
 }),

(BASE+"/LAB10/lab10.py", 10,
 "IPC PDF Chatbot using PyMuPDF",
 "Extracts text from an IPC PDF and answers keyword queries interactively using line-level search.",
 {
  1:  "Import fitz (PyMuPDF) for reading and parsing PDF files",
  4:  "Define extract(): reads every page of a PDF and returns all text",
  5:  "Docstring: Extract text from a PDF file",
  6:  "Initialise empty string to accumulate text from all pages",
  7:  "Open the PDF safely using a context manager (auto-closes on exit)",
  8:  "Iterate over every page object in the PDF document",
  9:  "Extract plain text from the page and append to the result string",
  10: "Return the complete concatenated text of the entire PDF",
  13: "Define search(): finds lines containing the query keyword",
  14: "Docstring: Search for a query in the IPC document",
  15: "Lowercase the query for case-insensitive comparison",
  16: "Split the full document text into individual lines",
  17: "Keep only lines where the lowercased query appears",
  19: "If matches found, return at most 15 lines to avoid overwhelming output",
  22: "Return a fallback message when no matching lines are found",
  26: "Define chatbot(): the main interactive query loop",
  27: "Docstring: Interactive chatbot for IPC document queries",
  28: "Notify user that the PDF is being loaded (may take a moment)",
  29: "Extract all text from the IPC PDF file at the given path",
  31: "Start an infinite loop to accept repeated user queries",
  32: "Prompt the user to type a question or keyword",
  33: "If the user types 'exit', break out of the loop",
  34: "Print a goodbye message on exit",
  37: "Search the extracted IPC text for the user's query",
  38: "Print all matching lines joined by newlines",
  39: "Print a separator line between consecutive query results",
  42: "Entry point: call chatbot() to start the interactive session",
 }),

]


def annotate_source(filepath, comments):
    """Read source file and return list of (line_no, original_line, comment) tuples."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            comment = comments.get(i, "")
            rows.append((i, line, comment))
    return rows


def build_pdf():
    doc = SimpleDocTemplate(OUTPUT, pagesize=A4,
                            leftMargin=14*mm, rightMargin=14*mm,
                            topMargin=14*mm, bottomMargin=14*mm)

    # ── Styles ──
    cover_h = ParagraphStyle("CH", fontSize=24, leading=30, alignment=TA_CENTER,
                              textColor=colors.HexColor("#0d47a1"), fontName="Helvetica-Bold", spaceAfter=6)
    cover_s = ParagraphStyle("CS", fontSize=12, leading=16, alignment=TA_CENTER,
                              textColor=colors.HexColor("#546e7a"), fontName="Helvetica", spaceAfter=4)
    note_s  = ParagraphStyle("NS", fontSize=9, alignment=TA_CENTER,
                              textColor=colors.grey, fontName="Helvetica-Oblique")
    lab_h   = ParagraphStyle("LH", fontSize=14, leading=18, textColor=colors.HexColor("#0d47a1"),
                              fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=2)
    desc_s  = ParagraphStyle("DS", fontSize=9, leading=13, textColor=colors.HexColor("#37474f"),
                              fontName="Helvetica-Oblique", spaceAfter=6)

    # Monospace styles
    code_plain = ParagraphStyle("CP", fontSize=8, leading=12, fontName="Courier",
                                 textColor=colors.HexColor("#1a237e"), leftIndent=0)
    code_comm  = ParagraphStyle("CC", fontSize=7.8, leading=12, fontName="Courier",
                                 textColor=colors.HexColor("#2e7d32"))

    story = []

    # ── Cover ──
    story.append(Spacer(1, 38*mm))
    story.append(Paragraph("Generative AI Lab", cover_h))
    story.append(Paragraph("Programs with Line-by-Line Explanations", cover_s))
    story.append(Paragraph("Labs 1 – 7 &nbsp;|&nbsp; Labs 9 – 10", cover_s))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("(Lab 8 excluded as per requirement)", note_s))
    story.append(PageBreak())

    for filepath, lab_num, title, desc, comments in LABS:
        story.append(Paragraph(f"Lab {lab_num}: {title}", lab_h))
        story.append(Paragraph(desc, desc_s))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#90caf9")))
        story.append(Spacer(1, 2*mm))

        rows = annotate_source(filepath, comments)

        for lineno, line, comment in rows:
            # Convert leading spaces to non-breaking spaces for indentation
            stripped = line.lstrip(" ")
            indent_count = len(line) - len(stripped)
            nbsp_indent = "\u00a0" * (indent_count * 1)  # 1 nbsp per space

            # Escape XML special chars
            def esc(s):
                return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            display_line = nbsp_indent + esc(stripped)

            if comment:
                # Line number + code + green inline comment
                full = f'<font color="#607d8b">{lineno:3d}│</font> {display_line}  <font color="#2e7d32"># {esc(comment)}</font>'
                story.append(Paragraph(full, code_plain))
            else:
                full = f'<font color="#607d8b">{lineno:3d}│</font> {display_line}'
                story.append(Paragraph(full, code_plain))

        story.append(Spacer(1, 4*mm))
        story.append(PageBreak())

    doc.build(story)
    print(f"✅  PDF saved → {OUTPUT}")


build_pdf()
