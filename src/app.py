import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from utils.words import analyze_sentiment, predict_emotion, preprocess_text
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
        return "URL tidak valid.", pd.DataFrame(), [], "❌ **Error:** URL YouTube tidak valid."

    iframe_html = f'<iframe width="100%" height="400" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>'
    comments_data = load_comments(video_id)
    
    total_comments = len(comments_data or [])
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
    # Jika data kosong, kembalikan figure kosong untuk kedua plot
    if not comments_data:
        fig_sent, ax_sent = plt.subplots()
        ax_sent.text(0.5, 0.5, "Tidak ada data", ha='center', va='center')
        fig_emo, ax_emo = plt.subplots()
        ax_emo.text(0.5, 0.5, "Tidak ada data", ha='center', va='center')
        return pd.DataFrame(), fig_sent, fig_emo

    results = []
    for comment in comments_data:
        text = comment.get("cleaned_text", "")
        
        if not text.strip():
            label_str = "Netral"
            confidence = 0.0
            emo_str = "Netral"
        else:
            result = analyze_sentiment(text)
            label_str = result[0]
            confidence = result[1]
            emo_str = predict_emotion(text)
        
        results.append({
            "author": comment["author"],
            "cleaned_text": text,
            "sentiment": label_str,
            "emotion": emo_str,
            "confidence": round(confidence, 4)
        })

    df_results = pd.DataFrame(results)

    # --- 1. Buat Visualisasi Distribusi Sentimen ---
    sentiment_counts = df_results['sentiment'].value_counts()
    color_map_sent = {"Positif": "#2ecc71", "Negatif": "#e74c3c", "Netral": "#95a5a6", "Error": "#000000"}
    colors_sent = [color_map_sent.get(idx, "#3498db") for idx in sentiment_counts.index]

    fig_sent, ax_sent = plt.subplots(figsize=(6, 4))
    bars_sent = sentiment_counts.plot(kind='bar', color=colors_sent, ax=ax_sent)
    ax_sent.set_title("Distribusi Sentimen (IndoBERT)", fontsize=12, fontweight='bold')
    ax_sent.set_ylabel("Jumlah Komentar", fontsize=10)
    ax_sent.set_xlabel("Sentimen", fontsize=10)
    ax_sent.tick_params(axis='x', rotation=0)
    
    for p in bars_sent.patches:
        ax_sent.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.tight_layout()

    # --- 2. Buat Visualisasi Distribusi Emosi ---
    emotion_counts = df_results['emotion'].value_counts()
    # Palet warna khusus untuk 5 label emosi
    color_map_emo = {
        "happy": "#f1c40f",   # Kuning
        "sadness": "#3498db", # Biru
        "anger": "#e74c3c",   # Merah
        "fear": "#9b59b6",    # Ungu
        "love": "#e84393"     # Pink
    }
    colors_emo = [color_map_emo.get(idx, "#1abc9c") for idx in emotion_counts.index]

    fig_emo, ax_emo = plt.subplots(figsize=(6, 4))
    bars_emo = emotion_counts.plot(kind='bar', color=colors_emo, ax=ax_emo)
    ax_emo.set_title("Distribusi Emosi (LSTM + FastText)", fontsize=12, fontweight='bold')
    ax_emo.set_ylabel("Jumlah Komentar", fontsize=10)
    ax_emo.set_xlabel("Emosi", fontsize=10)
    ax_emo.tick_params(axis='x', rotation=0) # Rotasi 0 jika teks muat, atau 45 jika kepanjangan
    
    for p in bars_emo.patches:
        ax_emo.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.tight_layout()

    return df_results, fig_sent, fig_emo

with gr.Blocks(title="Youtube Comment Sentiment Analysis") as app:
    gr.Markdown("# 📺 Youtube Comment Sentiment Analysis")

    with gr.Row():
        url_input = gr.Textbox(label="URL YouTube", placeholder=" (Example : https://www.youtube.com/watch?v=...)", lines=1)
        
    with gr.Row():
        load_btn = gr.Button("1. Load", variant="primary")

    with gr.Row():
        video_output = gr.HTML(label="Video Player", value=placeholder_html)

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
            # Header dataframe diperbarui untuk menampilkan kolom "emotion"
            sentiment_table = gr.Dataframe(headers=["author", "cleaned_text", "sentiment", "emotion", "confidence"], interactive=False, wrap=False)
        with gr.Column(scale=1):
            # Dua komponen plot ditumpuk secara vertikal di kolom sebelah kanan
            sentiment_plot = gr.Plot(label="Grafik Sentimen")
            emotion_plot = gr.Plot(label="Grafik Emosi")

    raw_comments_state = gr.State([])
    clean_comments_state = gr.State([])

    load_btn.click(
        fn=load_video_and_comments,
        inputs=[url_input],
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
        # Tambahkan output emotion_plot ke listener
        outputs=[sentiment_table, sentiment_plot, emotion_plot]
    )

if __name__ == "__main__":
    app.launch()