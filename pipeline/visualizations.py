import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


def save_pyldavis(lda, corpus, dictionary, f_name):
    vis = gensimvis.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(vis, f_name)
    print(f"Saved: {f_name}")


def save_topic_words_html(lda_model, f_name, label):
    html = f"""
            <html>
              <body>
                 <h1>{label}</h1>
           """

    for topic_id, terms in lda_model.show_topics(num_words=20, formatted=False):
        html += f"<h2>{topic_id}</h2>"
        for word, prob in terms:
            bar = int(prob * 100)
            html += f"<div><b>{word}</b>{prob:.4f}</div>"
            html += (f"<div style='height:20px;width:{bar}px;"
                     f"background-color:steelblue'></div>")
    html += f"</body></html>"

    with open(f_name, "w") as f:
        f.write(html)

    print("Saved:", f_name)


def save_compare_html():
    compare_html = """
        <html>
            <body>
                <h1>LDA Comparison</h1>
                    <iframe src="lda_original.html" width="49%" height="1000px"></iframe>
                    <iframe src="lda_resolved.html" width="49%" height="1000px"></iframe>
            </body>
        </html>
    """

    with open("lda_comparision.html", "w") as f:
        f.write(compare_html)

    print("Saved: lda_comparision.html")


