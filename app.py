import pandas as pd
import streamlit as st

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# SAYFA AYARLARI
# ============================================================

st.set_page_config(
    page_title="Müşteri Yorumları Duygu Analizi",
    page_icon="💬",
    layout="wide"
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

    df = pd.read_csv(
        DATA_PATH,
        sep=";"
    )

    df = df.dropna(
        subset=["Label", "Text"]
    ).copy()

    df["Label"] = df["Label"].astype(int)
    df["Text"] = df["Text"].astype(str)

    X = df["Text"]
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # TF-IDF
    # Tek kelimeler + iki kelimelik kalıplar
    # --------------------------------------------------------

    tfidf = TfidfVectorizer(
        lowercase=True,
        max_features=7000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = tfidf.fit_transform(
        X_train
    )

    X_test_tfidf = tfidf.transform(
        X_test
    )

    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    y_pred = model.predict(
        X_test_tfidf
    )

    metrics = {

        "accuracy": accuracy_score(
            y_test,
            y_pred
        ),

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

        "feature_count": X_train_tfidf.shape[1]
    }

    return df, model, tfidf, metrics


# ============================================================
# TOPLU YORUM ANALİZİ
# ============================================================

def yorumlari_analiz_et(
    yorum_df,
    model,
    tfidf
):

    sonuc_df = yorum_df.copy()

    sonuc_df["Text"] = sonuc_df[
        "Text"
    ].astype(str)

    yorum_tfidf = tfidf.transform(
        sonuc_df["Text"]
    )

    tahminler = model.predict(
        yorum_tfidf
    )

    olasiliklar = model.predict_proba(
        yorum_tfidf
    )

    duygu_listesi = []
    oran_listesi = []

    for i in range(
        len(tahminler)
    ):

        if tahminler[i] == 1:

            duygu = "Olumlu"

            oran = (
                olasiliklar[i][1]
                * 100
            )

        else:

            duygu = "Olumsuz"

            oran = (
                olasiliklar[i][0]
                * 100
            )

        duygu_listesi.append(
            duygu
        )

        oran_listesi.append(
            round(
                oran,
                1
            )
        )

    sonuc_df[
        "Duygu"
    ] = duygu_listesi

    sonuc_df[
        "Tahmin_Olasiligi"
    ] = oran_listesi

    return sonuc_df


# ============================================================
# CSV OKUMA FONKSİYONU
# ============================================================

def csv_dosyasini_oku(
    uploaded_file
):

    denemeler = [

        {
            "encoding": "utf-8-sig",
            "sep": ","
        },

        {
            "encoding": "utf-8",
            "sep": ","
        },

        {
            "encoding": "utf-8-sig",
            "sep": ";"
        },

        {
            "encoding": "utf-8",
            "sep": ";"
        },

        {
            "encoding": "cp1254",
            "sep": ";"
        },

        {
            "encoding": "cp1254",
            "sep": ","
        }
    ]

    for ayar in denemeler:

        try:

            uploaded_file.seek(0)

            temp_df = pd.read_csv(
                uploaded_file,
                encoding=ayar[
                    "encoding"
                ],
                sep=ayar[
                    "sep"
                ]
            )

            if "Text" in temp_df.columns:

                return temp_df

        except Exception:

            pass

    return None


# ============================================================
# MODELİ BAŞLAT
# ============================================================

try:

    df, model, tfidf, metrics = modeli_hazirla()

except Exception as hata:

    st.error(
        f"Model başlatılamadı: {hata}"
    )

    st.stop()


# ============================================================
# BAŞLIK
# ============================================================

st.title(
    "💬 Müşteri Yorumları Duygu Analizi"
)

st.write(
    "Türkçe müşteri yorumlarını "
    "**TF-IDF + Logistic Regression** "
    "kullanarak olumlu veya olumsuz "
    "olarak sınıflandıran makine öğrenmesi uygulaması."
)

st.divider()


# ============================================================
# KPI KARTLARI
# ============================================================

col1, col2, col3, col4 = st.columns(
    4
)

col1.metric(
    "Toplam Veri",
    metrics["total_count"]
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
    metrics["feature_count"]
)

st.caption(
    f"Model {metrics['train_count']} eğitim "
    f"ve {metrics['test_count']} test yorumu "
    f"ile değerlendirildi."
)

st.divider()


# ============================================================
# SEKMELER
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Proje Özeti",
        "✍️ Tek Yorum Analizi",
        "📁 Toplu Yorum Analizi"
    ]
)


# ============================================================
# TAB 1
# PROJE ÖZETİ
# ============================================================

with tab1:

    st.subheader(
        "Model Nasıl Çalışıyor?"
    )

    st.info(
        "Müşteri Yorumu → TF-IDF → "
        "Logistic Regression → "
        "Olumlu / Olumsuz Tahmini"
    )

    st.write(
        "TF-IDF, yorum metinlerini modelin "
        "işleyebileceği sayısal özelliklere dönüştürür. "
        "Logistic Regression modeli ise bu özellikleri "
        "kullanarak yorumun olumlu veya olumsuz "
        "olma olasılığını tahmin eder."
    )

    st.subheader(
        "Model Performansı"
    )

    performans_df = pd.DataFrame(
        {

            "Metrik": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
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
                )
            ]
        }
    )

    st.dataframe(
        performans_df,
        hide_index=True,
        use_container_width=True
    )

    st.subheader(
        "Veri Seti Duygu Dağılımı"
    )

    olumlu_sayisi = int(
        (
            df["Label"] == 1
        ).sum()
    )

    olumsuz_sayisi = int(
        (
            df["Label"] == 0
        ).sum()
    )

    dagilim_df = pd.DataFrame(
        {

            "Duygu": [
                "Olumlu",
                "Olumsuz"
            ],

            "Yorum Sayısı": [
                olumlu_sayisi,
                olumsuz_sayisi
            ]
        }
    )

    st.bar_chart(
        dagilim_df.set_index(
            "Duygu"
        )
    )


# ============================================================
# TAB 2
# TEK YORUM ANALİZİ
# ============================================================

with tab2:

    st.subheader(
        "Tek Yorum Analizi"
    )

    st.write(
        "Bir müşteri yorumu yazın ve "
        "**Yorumu Analiz Et** butonuna basın."
    )

    with st.form(
        "yorum_formu",
        clear_on_submit=False
    ):

        yorum = st.text_area(
            "Müşteri Yorumu",
            placeholder=(
                "Örnek: Ürün çok kaliteli, "
                "hızlı geldi ve çok memnun kaldım."
            ),
            height=160
        )

        analiz_et = st.form_submit_button(
            "🔍 Yorumu Analiz Et",
            type="primary",
            use_container_width=True
        )


    if analiz_et:

        if yorum.strip() == "":

            st.warning(
                "Lütfen bir yorum yazın."
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

            if tahmin == 1:

                sonuc = "Olumlu"

                oran = float(
                    olasiliklar[1]
                    * 100
                )

            else:

                sonuc = "Olumsuz"

                oran = float(
                    olasiliklar[0]
                    * 100
                )

            st.session_state[
                "son_yorum"
            ] = yorum

            st.session_state[
                "son_sonuc"
            ] = sonuc

            st.session_state[
                "son_oran"
            ] = oran


    if "son_sonuc" in st.session_state:

        st.divider()

        st.write(
            "**Analiz edilen yorum:**"
        )

        st.write(
            st.session_state[
                "son_yorum"
            ]
        )

        if (
            st.session_state[
                "son_sonuc"
            ]
            == "Olumlu"
        ):

            st.success(
                "😊 OLUMLU YORUM"
            )

            st.metric(
                "Olumlu Tahmin Olasılığı",
                f"%{st.session_state['son_oran']:.1f}"
            )

        else:

            st.error(
                "😞 OLUMSUZ YORUM"
            )

            st.metric(
                "Olumsuz Tahmin Olasılığı",
                f"%{st.session_state['son_oran']:.1f}"
            )

        progress_degeri = int(
            round(
                st.session_state[
                    "son_oran"
                ]
            )
        )

        progress_degeri = max(
            0,
            min(
                progress_degeri,
                100
            )
        )

        st.progress(
            progress_degeri
        )

        st.caption(
            "Gösterilen oran, modelin tahmin ettiği "
            "sınıfa ait olasılıktır."
        )


# ============================================================
# TAB 3
# TOPLU YORUM ANALİZİ
# ============================================================

with tab3:

    st.subheader(
        "CSV ile Toplu Yorum Analizi"
    )

    st.write(
        "CSV dosyasında müşteri yorumlarının "
        "bulunduğu sütunun adı **Text** olmalıdır."
    )


    # --------------------------------------------------------
    # DEMO CSV
    # --------------------------------------------------------

    demo_df = pd.DataFrame(
        {

            "Text": [

                "Ürün çok güzel ve kaliteli, çok memnun kaldım.",

                "Kargo çok geç geldi ve paket hasarlıydı.",

                "Beklediğimden daha iyi çıktı, teşekkür ederim.",

                "Ürün çalışmıyor, iade etmek istiyorum.",

                "Hızlı teslimat ve güzel paketleme.",

                "Kalitesi çok kötü, hiç memnun kalmadım.",

                "Fiyatına göre oldukça başarılı.",

                "Eksik ürün gönderilmiş, hayal kırıklığı yaşadım.",

                "Çok beğendim, tekrar satın alırım.",

                "Teslimat çok gecikti."
            ]
        }
    )

    demo_csv = demo_df.to_csv(
        index=False
    ).encode(
        "utf-8-sig"
    )

    st.download_button(
        "⬇️ Örnek CSV Dosyasını İndir",
        data=demo_csv,
        file_name="demo_musteri_yorumlari.csv",
        mime="text/csv"
    )

    st.divider()


    # --------------------------------------------------------
    # DOSYA YÜKLE
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "CSV Dosyası Seçin",
        type=["csv"],
        accept_multiple_files=False,
        key="csv_file_uploader"
    )


    # --------------------------------------------------------
    # DOSYA SEÇİLDİ Mİ?
    # --------------------------------------------------------

    if uploaded_file is None:

        st.info(
            "Henüz CSV dosyası seçilmedi."
        )

    else:

        st.success(
            f"✅ Dosya yüklendi: {uploaded_file.name}"
        )

        st.write(
            f"Dosya boyutu: "
            f"{uploaded_file.size / 1024:.1f} KB"
        )

        analiz_baslat = st.button(
            "🚀 Toplu Analizi Başlat",
            type="primary",
            use_container_width=True,
            key="toplu_analiz_butonu"
        )


        # ----------------------------------------------------
        # ANALİZ BUTONU
        # ----------------------------------------------------

        if analiz_baslat:

            yeni_df = csv_dosyasini_oku(
                uploaded_file
            )


            # ------------------------------------------------
            # DOSYA OKUNAMADI
            # ------------------------------------------------

            if yeni_df is None:

                st.error(
                    "CSV dosyası okunamadı veya "
                    "'Text' sütunu bulunamadı."
                )

                st.write(
                    "CSV dosyasının ilk sütununun adının "
                    "**Text** olduğundan emin olun."
                )


            # ------------------------------------------------
            # DOSYA OKUNDU
            # ------------------------------------------------

            else:

                yeni_df = yeni_df.dropna(
                    subset=["Text"]
                ).copy()

                if yeni_df.empty:

                    st.warning(
                        "Dosyada analiz edilecek "
                        "yorum bulunamadı."
                    )

                else:

                    sonuc_df = yorumlari_analiz_et(
                        yeni_df,
                        model,
                        tfidf
                    )

                    st.session_state[
                        "toplu_sonuc_df"
                    ] = sonuc_df


    # ========================================================
    # SONUÇLARI GÖSTER
    # ========================================================

    if (
        "toplu_sonuc_df"
        in st.session_state
    ):

        sonuc_df = st.session_state[
            "toplu_sonuc_df"
        ]

        st.divider()

        st.success(
            "✅ Toplu analiz tamamlandı."
        )

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
            olumlu
            / toplam
            * 100
        )

        olumsuz_oran = (
            olumsuz
            / toplam
            * 100
        )


        # ----------------------------------------------------
        # KPI KARTLARI
        # ----------------------------------------------------

        k1, k2, k3 = st.columns(
            3
        )

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

        st.subheader(
            "Duygu Dağılımı"
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
                ]
            }
        )

        st.bar_chart(
            grafik_df.set_index(
                "Duygu"
            )
        )


        # ----------------------------------------------------
        # SONUÇ TABLOSU
        # ----------------------------------------------------

        st.subheader(
            "Analiz Sonuçları"
        )

        filtre = st.selectbox(
            "Yorumları Filtrele",
            [
                "Tümü",
                "Olumlu",
                "Olumsuz"
            ],
            key="sonuc_filtre"
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
            hide_index=True
        )


        # ----------------------------------------------------
        # ÖNCELİKLİ OLUMSUZ YORUMLAR
        # ----------------------------------------------------

        st.subheader(
            "Öncelikli Olumsuz Yorumlar"
        )

        kritik_df = (

            sonuc_df[
                sonuc_df["Duygu"]
                == "Olumsuz"
            ]

            .sort_values(
                "Tahmin_Olasiligi",
                ascending=False
            )

            .head(5)
        )

        if kritik_df.empty:

            st.info(
                "Olumsuz olarak sınıflandırılan "
                "yorum bulunmadı."
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
                hide_index=True
            )


        # ----------------------------------------------------
        # CSV İNDİR
        # ----------------------------------------------------

        sonuc_csv = sonuc_df.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        )

        st.download_button(
            "⬇️ Analiz Sonuçlarını CSV Olarak İndir",
            data=sonuc_csv,
            file_name="duygu_analizi_sonuclari.csv",
            mime="text/csv"
        )


# ============================================================
# ALT BİLGİ
# ============================================================

st.divider()

st.caption(
    "Python • NLP • TF-IDF • "
    "Logistic Regression • Streamlit"
)

st.caption(
    "Eğitim ve portföy amacıyla hazırlanmıştır."
)