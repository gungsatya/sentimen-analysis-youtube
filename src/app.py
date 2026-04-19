import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from utils.words import analyze_sentiment, preprocess_text
from utils.youtube import get_video_id, load_comments

placeholder_html = """
<div style="
    width: 100%; 
    height: 400px; 
    background-color: #f3f4f6; 
    border: 2px dashed #9ca3af; 
    border-radius: 10px; 
    display: flex; 
    flex-direction: column;
    align-items: center; 
    justify-content: center; 
    color: #4b5563; 
    font-family: sans-serif;
">
    <div style="font-size: 40px; margin-bottom: 10px;">📺</div>
    <h3 style="margin: 0;">Video YouTube Akan Tampil Di Sini</h3>
    <p style="margin-top: 5px; font-size: 14px; color: #6b7280;">Masukkan URL dan klik "Load" untuk memulai</p>
</div>
"""



def load_video_and_comments(url):
    video_id = get_video_id(url)
    if not video_id:
        # Kembalikan pesan error pada indikator status
        return "URL tidak valid.", pd.DataFrame(), [], "❌ **Error:** URL YouTube tidak valid."

    iframe_html = f'<iframe width="100%" height="400" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>'

    comments_data = load_comments(video_id)
    
    # Hitung jumlah komentar yang ditarik
    total_comments = len(comments_data or [])
    
    # Buat teks indikator
    status_text = f"✅ **Berhasil memuat {total_comments} komentar.**"
   
    return iframe_html, pd.DataFrame(comments_data), comments_data, status_text

def preprocess_comments(comments):
    cleaned_comments = []
    
    for comment in comments:
        cleaned_text = preprocess_text(comment['raw_text'])
        cleaned_comments.append({
            'author': comment['author'],
            'cleaned_text': cleaned_text,
            'raw_text': comment['raw_text'],
            'uuid': comment['uuid']
        })

    return pd.DataFrame(cleaned_comments), cleaned_comments

def process_sentiment(comments_data):
    if not comments_data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Tidak ada data", ha='center', va='center')
        return pd.DataFrame(), fig

    results = []
    for comment in comments_data:
        text = comment.get("cleaned_text", "")
        
        if not text.strip():
            label_str = "Netral"
            confidence = 0.0
        else:
            # Batasi panjang teks untuk menghindari error max_token BERT (512 tokens)
            # 1500 karakter biasanya aman untuk limit 512 subword token
            result = analyze_sentiment(text)
            label_str = result[0]
            confidence = result[1]
        
        results.append({
            "author": comment["author"],
            "cleaned_text": text,
            "sentiment": label_str,
            "confidence": round(confidence, 4)
        })

    df_results = pd.DataFrame(results)

    # Buat Visualisasi Distribusi
    sentiment_counts = df_results['sentiment'].value_counts()
    color_map = {"Positif": "#2ecc71", "Negatif": "#e74c3c", "Netral": "#95a5a6", "Error": "#000000"}
    colors = [color_map.get(idx, "#3498db") for idx in sentiment_counts.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = sentiment_counts.plot(kind='bar', color=colors, ax=ax)
    
    ax.set_title("Distribusi Sentimen (IndoBERT)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Jumlah Komentar", fontsize=10)
    ax.set_xlabel("Sentimen", fontsize=10)
    ax.tick_params(axis='x', rotation=0)
    
    for p in bars.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
        
    plt.tight_layout()

    return df_results, fig

with gr.Blocks(title="Youtube Comment Sentiment Analysis") as app:
    gr.Markdown("# 📺 Youtube Comment Sentiment Analysis")

    with gr.Row():
        url_input = gr.Textbox(label="URL YouTube", placeholder=" (Example : https://www.youtube.com/watch?v=...)", lines=1)
        
    with gr.Row():
        load_btn = gr.Button("1. Load", variant="primary")

    with gr.Row():
        video_output = gr.HTML(label="Video Player", value=placeholder_html)

    # Menambahkan indikator status di sini
    with gr.Row():
        status_indicator = gr.Markdown("⏳ *Belum ada data yang dimuat.*")

    with gr.Row():
        raw_comments_table = gr.Dataframe(label="Comments", headers=["author", "raw_text", "uuid"], interactive=False, wrap=True,)

    with gr.Row():
        preprocess_btn = gr.Button("2. Preprocess", variant="secondary")

    with gr.Row():
        cleaned_comments_table = gr.Dataframe(label="Cleaned Comments", headers=["author", "cleaned_text", "raw_text", "uuid"], interactive=False, wrap=True,)
    
    with gr.Row():
        process_btn = gr.Button("3. Process", variant="secondary")

    gr.Markdown("### Results")
    with gr.Row():
        with gr.Column(scale=2):
            sentiment_table = gr.Dataframe(headers=["author", "cleaned_text", "sentiment", "confidence"], interactive=False, wrap=False)
        with gr.Column(scale=1):
            sentiment_plot = gr.Plot(label="Grafik Sentimen")

    # State untuk menyimpan data antar proses
    raw_comments_state = gr.State([])
    clean_comments_state = gr.State([])

    # Event Listeners
    load_btn.click(
        fn=load_video_and_comments,
        inputs=[url_input],
        # Tambahkan status_indicator ke dalam outputs
        outputs=[video_output, raw_comments_table, raw_comments_state, status_indicator]
    )

    preprocess_btn.click(
        fn=preprocess_comments,
        inputs=[raw_comments_state],
        outputs=[cleaned_comments_table, clean_comments_state]
    )

    process_btn.click(
        fn=process_sentiment,
        inputs=[clean_comments_state],
        outputs=[sentiment_table, sentiment_plot]
    )

if __name__ == "__main__":
    app.launch()