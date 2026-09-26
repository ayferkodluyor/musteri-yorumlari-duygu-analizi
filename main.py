from pathlib import Path

import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# SAYFA AYARLARI
# ============================================================

st.set_page_config(
    page_title="Müşteri Yorumları Duygu Analizi",
    page_icon="💬",
    layout="wide",
)

# ============================================================
# GÖRSEL DÜZEN
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        .hero {
            padding: 1.6rem 1.8rem;
            border-radius: 18px;
            border: 1px solid rgba(128,128,128,0.20);
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin-bottom: 0.4rem;
            font-size: 2.15rem;
        }

        .hero p {
            margin: 0;
            font-size: 1.02rem;
            opacity: 0.85;
        }

        .small-note {
            opacity: 0.75;
            font-size: 0.90rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,0.18);
            padding: 0.9rem;
            border-radius: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DOSYA YOLU
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "sample2000binary.csv"

# ============================================================
# MODELİ HAZIRLA
# ============================================================

@st.cache_resource
def modeli_hazirla():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "sample2000binary.csv dosyası app.py ile aynı klasörde bulunamadı."
        )

    df = pd.read_csv(DATA_PATH, sep=";")
    df = df.dropna(subset=["Label", "Text"]).copy()

    # True = 1 (Olumlu)
    # False = 0 (Olumsuz)
    df["Label"] = df["Label"].astype(int)
    df["Text"] = df["Text"].astype(str)

    X = df["Text"]
    y = df["Label"]

    # %75 eğitim - %25 test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Metni sayısal özelliklere dönüştür
    tfidf = TfidfVectorizer(
        lowercase=True,
        max_features=5000,
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Model
    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    # Test tahmini
    y_pred = model.predict(
        X_test_tfidf
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "train_count": len(X_train),
        "test_count": len(X_test),
        "total_count": len(df),
        "feature_count": X_train_tfidf.shape[1],
    }

    return df, model, tfidf, metrics


# ============================================================
# TOPLU YORUM ANALİZ FONKSİYONU
# ============================================================

def yorumlari_analiz_et(
    df_yorum,
    model,
    tfidf
):

    sonuc = df_yorum.copy()

    sonuc["Text"] = sonuc["Text"].astype(str)

    X_new = tfidf.transform(
        sonuc["Text"]
    )

    tahminler = model.predict(
        X_new
    )

    olasiliklar = model.predict_proba(
        X_new
    )

    sonuc["Duygu"] = [
        "Olumlu" if tahmin == 1 else "Olumsuz"
        for tahmin in tahminler
    ]

    sonuc["Tahmin_Olasiligi"] = [
        round(
            (
                olasiliklar[i][1]
                if tahminler[i] == 1
                else olasiliklar[i][0]
            ) * 100,
            1,
        )
        for i in range(len(tahminler))
    ]

    return sonuc


# ============================================================
# MODELİ ÇALIŞTIR
# ============================================================

try:

    df, model, tfidf, metrics = modeli_hazirla()

except Exception as e:

    st.error(
        f"Uygulama başlatılamadı: {e}"
    )

    st.stop()


# ============================================================
# BAŞLIK
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>💬 Müşteri Yorumları Duygu Analizi</h1>

        <p>
            Türkçe müşteri yorumlarını
            TF-IDF ve Logistic Regression kullanarak
            olumlu veya olumsuz olarak sınıflandıran
            Streamlit web uygulaması.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ÜST KPI KARTLARI
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Veri Seti",
    f"{metrics['total_count']:,}".replace(",", ".")
)

col2.metric(
    "Test Accuracy",
    f"%{metrics['accuracy'] * 100:.1f}"
)

col3.metric(
    "F1 Score",
    f"{metrics['f1']:.2f}"
)

col4.metric(
    "TF-IDF Özelliği",
    f"{metrics['feature_count']:,}".replace(",", ".")
)

st.caption(
    f"Model {metrics['train_count']} eğitim "
    f"ve {metrics['test_count']} test yorumu ile değerlendirildi."
)


# ============================================================
# SEKMELER
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🏠 Proje Özeti",
        "✍️ Tek Yorum Analizi",
        "📁 Toplu Yorum Analizi",
    ]
)


# ============================================================
# TAB 1 - PROJE ÖZETİ
# ============================================================

with tab1:

    st.subheader(
        "Proje Akışı"
    )

    st.info(
        "Yorum metni → "
        "TF-IDF ile sayısal özellikler → "
        "Logistic Regression → "
        "Olumlu / Olumsuz tahmini"
    )

    a, b = st.columns(2)

    # --------------------------------------------------------
    # MODEL PERFORMANSI
    # --------------------------------------------------------

    with a:

        st.markdown(
            "#### Model Performansı"
        )

        performans_df = pd.DataFrame(
            {
                "Metrik": [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                ],

                "Değer": [
                    round(
                        metrics["accuracy"],
                        3
                    ),
                    round(
                        metrics["precision"],
                        3
                    ),
                    round(
                        metrics["recall"],
                        3
                    ),
                    round(
                        metrics["f1"],
                        3
                    ),
                ],
            }
        )

        st.dataframe(
            performans_df,
            hide_index=True,
            use_container_width=True
        )

    # --------------------------------------------------------
    # VERİ DAĞILIMI
    # --------------------------------------------------------

    with b:

        st.markdown(
            "#### Veri Dağılımı"
        )

        dagilim = (
            df["Label"]
            .map(
                {
                    1: "Olumlu",
                    0: "Olumsuz"
                }
            )
            .value_counts()
            .rename_axis("Duygu")
            .reset_index(
                name="Yorum Sayısı"
            )
        )

        st.bar_chart(
            dagilim.set_index(
                "Duygu"
            )
        )

    st.markdown(
        """
        <p class="small-note">
        Not: Bu uygulama eğitim ve portföy amacıyla hazırlanmıştır.
        Model sonuçları otomatik sınıflandırma tahminidir.
        </p>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TAB 2 - TEK YORUM ANALİZİ
# ============================================================

with tab2:

    st.subheader(
        "Tek Bir Yorumu Analiz Et"
    )

    yorum = st.text_area(
        "Müşteri yorumunu yazın",
        placeholder=(
            "Örn: Ürün çok kaliteli, "
            "hızlı geldi ve çok memnun kaldım."
        ),
        height=150,
        key="tek_yorum",
    )

    analiz_butonu = st.button(
        "🔍 Yorumu Analiz Et",
        type="primary",
        use_container_width=True,
    )

    if analiz_butonu:

        if yorum.strip() == "":

            st.warning(
                "Lütfen analiz etmek için bir yorum yazın."
            )

        else:

            yorum_tfidf = tfidf.transform(
                [yorum]
            )

            tahmin = int(
                model.predict(
                    yorum_tfidf
                )[0]
            )

            olasiliklar = model.predict_proba(
                yorum_tfidf
            )[0]

            # ------------------------------------------------
            # OLUMLU
            # ------------------------------------------------

            if tahmin == 1:

                oran = float(
                    olasiliklar[1] * 100
                )

                st.success(
                    "😊 OLUMLU YORUM"
                )

                st.metric(
                    "Olumlu Tahmin Olasılığı",
                    f"%{oran:.1f}"
                )

            # ------------------------------------------------
            # OLUMSUZ
            # ------------------------------------------------

            else:

                oran = float(
                    olasiliklar[0] * 100
                )

                st.error(
                    "😞 OLUMSUZ YORUM"
                )

                st.metric(
                    "Olumsuz Tahmin Olasılığı",
                    f"%{oran:.1f}"
                )

            st.progress(
                min(
                    max(
                        oran / 100,
                        0.0
                    ),
                    1.0
                )
            )

            st.caption(
                "Gösterilen oran modelin tahmin ettiği "
                "sınıfa ait olasılıktır; "
                "gerçek müşteri duygusunun kesin ölçümü değildir."
            )


# ============================================================
# TAB 3 - TOPLU ANALİZ
# ============================================================

with tab3:

    st.subheader(
        "CSV ile Toplu Yorum Analizi"
    )

    st.write(
        "CSV dosyanızda yorumların bulunduğu "
        "sütunun adı **Text** olmalıdır."
    )

    # --------------------------------------------------------
    # DEMO DOSYASI
    # --------------------------------------------------------

    demo_df = pd.DataFrame(
        {
            "Text": [

                "Ürün çok güzel ve kaliteli, çok memnun kaldım.",

                "Kargo çok geç geldi ve paket hasarlıydı.",

                "Beklediğimden daha iyi çıktı, teşekkür ederim.",

                "Ürün çalışmıyor, iade etmek istiyorum.",

                "Hızlı teslimat ve güzel paketleme için teşekkürler.",

                "Kalitesi çok kötü, hiç memnun kalmadım.",

                "Fiyatına göre gayet başarılı bir ürün.",

                "Eksik ürün gönderilmiş, büyük hayal kırıklığı.",

                "Çok beğendim, tekrar satın alırım.",

                "Teslimat gecikti ve müşteri hizmetlerine ulaşamadım.",
            ]
        }
    )

    st.download_button(
        "⬇️ Demo CSV Dosyasını İndir",
        data=demo_df.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        ),
        file_name="demo_musteri_yorumlari.csv",
        mime="text/csv",
    )

    # --------------------------------------------------------
    # CSV YÜKLE
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Analiz edilecek CSV dosyasını seçin",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:

            uploaded_file.seek(0)

            yeni_df = pd.read_csv(
                uploaded_file,
                sep=None,
                engine="python",
            )

        except Exception as e:

            st.error(
                f"CSV dosyası okunamadı: {e}"
            )

            st.stop()

        # ----------------------------------------------------
        # TEXT SÜTUNU KONTROL
        # ----------------------------------------------------

        if "Text" not in yeni_df.columns:

            st.error(
                "Dosyada 'Text' isimli sütun bulunamadı. "
                "Yorum sütununun adını Text yapıp tekrar yükleyin."
            )

        else:

            yeni_df = yeni_df.dropna(
                subset=["Text"]
            ).copy()

            if len(yeni_df) == 0:

                st.warning(
                    "CSV dosyasında analiz edilecek yorum bulunamadı."
                )

            else:

                sonuc_df = yorumlari_analiz_et(
                    yeni_df,
                    model,
                    tfidf,
                )

                st.session_state[
                    "toplu_sonuc"
                ] = sonuc_df


    # ========================================================
    # TOPLU ANALİZ SONUÇLARI
    # ========================================================

    if "toplu_sonuc" in st.session_state:

        sonuc_df = st.session_state[
            "toplu_sonuc"
        ]

        toplam = len(
            sonuc_df
        )

        olumlu = int(
            (
                sonuc_df["Duygu"]
                == "Olumlu"
            ).sum()
        )

        olumsuz = int(
            (
                sonuc_df["Duygu"]
                == "Olumsuz"
            ).sum()
        )

        olumlu_oran = (
            olumlu / toplam
        ) * 100

        olumsuz_oran = (
            olumsuz / toplam
        ) * 100

        st.success(
            "Toplu analiz tamamlandı."
        )

        # ----------------------------------------------------
        # KPI KARTLARI
        # ----------------------------------------------------

        k1, k2, k3 = st.columns(3)

        k1.metric(
            "Toplam Yorum",
            toplam
        )

        k2.metric(
            "😊 Olumlu",
            f"{olumlu} (%{olumlu_oran:.1f})"
        )

        k3.metric(
            "😞 Olumsuz",
            f"{olumsuz} (%{olumsuz_oran:.1f})"
        )

        # ----------------------------------------------------
        # GRAFİK
        # ----------------------------------------------------

        st.markdown(
            "#### Duygu Dağılımı"
        )

        grafik_df = pd.DataFrame(
            {
                "Duygu": [
                    "Olumlu",
                    "Olumsuz"
                ],

                "Yorum Sayısı": [
                    olumlu,
                    olumsuz
                ],
            }
        )

        st.bar_chart(
            grafik_df.set_index(
                "Duygu"
            )
        )

        # ----------------------------------------------------
        # TABLO
        # ----------------------------------------------------

        st.markdown(
            "#### Analiz Edilen Yorumlar"
        )

        filtre = st.selectbox(
            "Gösterilecek yorumlar",
            [
                "Tümü",
                "Olumlu",
                "Olumsuz"
            ],
        )

        if filtre == "Tümü":

            gosterilecek_df = sonuc_df

        else:

            gosterilecek_df = sonuc_df[
                sonuc_df["Duygu"]
                == filtre
            ]

        st.dataframe(
            gosterilecek_df,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # ÖNCELİKLİ OLUMSUZ YORUMLAR
        # ----------------------------------------------------

        st.markdown(
            "#### Öncelikli Olumsuz Yorumlar"
        )

        kritik_df = (
            sonuc_df[
                sonuc_df["Duygu"]
                == "Olumsuz"
            ]
            .sort_values(
                "Tahmin_Olasiligi",
                ascending=False,
            )
            .head(5)
        )

        if len(kritik_df) == 0:

            st.info(
                "Bu dosyada olumsuz olarak "
                "sınıflandırılan yorum bulunmadı."
            )

        else:

            st.dataframe(
                kritik_df[
                    [
                        "Text",
                        "Tahmin_Olasiligi"
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # SONUÇLARI İNDİR
        # ----------------------------------------------------

        st.download_button(
            "⬇️ Analiz Sonuçlarını CSV Olarak İndir",

            data=sonuc_df.to_csv(
                index=False
            ).encode(
                "utf-8-sig"
            ),

            file_name=(
                "duygu_analizi_sonuclari.csv"
            ),

            mime="text/csv",
        )


# ============================================================
# ALT BİLGİ
# ============================================================

st.divider()

st.caption(
    "Python • NLP • TF-IDF • Logistic Regression • Streamlit "
    "| Eğitim ve portföy projesi"
)