import pandas as pd
import yfinance as yf
import streamlit as st
from PIL import Image
import datetime
from datetime import datetime, date,time,timedelta
import matplotlib.pyplot as plt
import prophet
from prophet import Prophet
import seaborn as sns
import base64
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric

st.title("Beytem Finansal Analiz")
st.write("Datan Analizi ve Görselleştirme Paneli. Bu sayfadaki veriler yatırım tavsiyesi değildir.")

st.sidebar.title("Filtrele")
st.markdown("<a href='https://www.beytem.com.tr'> BEYTEM </a>", unsafe_allow_html=True)

# data=yf.Ticker("ETH-USD")
# df=data.history(period="Id",start="2010-5-21",end="2023-1-28")
# df.columns=("Açılış","En Yüksek","En Düşük","Kapanış","Hacim","Temettüler","Bölünmüş")
# st.dataframe(df)
# st.line_chart(df["Kapanış"])
# st.line_chart(df["Hacim"])
islemturu = st.sidebar.radio("İşlem Türü", ["Kripto", "Borsa"])
if islemturu == "Kripto":
    kriptosec = st.sidebar.selectbox("Kripto Para Cinsi", ["BTC", "ETH", "XRP", "DOT", "DOGE", "AVAX", "BNB"])
    kriptosec = kriptosec+"-USD"
    sembol=kriptosec

else:
    borsasec = st.sidebar.selectbox("Hisse Senetleri", ["ASELSAN", "THY", "GARANTİ", "AKBANK", "BJK"])
    senetler = {
        "ASELSAN": "ASELS.IS",
        "THY": "THYAO.IS",
        "GARANTİ": "GARAN.IS",
        "AKBANK": "AKBNK.IS",
        "BJK": "BJKAS.IS"
    }
    hissesec = senetler[borsasec]
    sembol = hissesec

zaralik=range(1,721)
slider=st.sidebar.select_slider("Zaman Aralığı",options=zaralik,value=30)

bugun=datetime.today()
aralik=timedelta(days=slider)
st.sidebar.write("### Tarih Aralığı")
baslangic=st.sidebar.date_input("Başlangıç Tarihi",value=bugun-aralik)
bitis=st.sidebar.date_input("Bitiş Tarihi",value=bugun)
st.sidebar.write("### Machine Learning Tahmin")
prophet=st.sidebar.checkbox("ML Prophet Tahmini")
if prophet:
    fbaralik = range(1, 1441)
    fbperiyot = st.sidebar.select_slider("Periyot", options=zaralik, value=30)
    components=st.sidebar.checkbox("Components")
global model,cvin,period,horizon,metric





def grafikgetir(sembol,baslangic,bitis):
    data = yf.Ticker(sembol)
    global df
    st.write("Veri Tablosu")
    df=data.history(period="Id",start=baslangic,end=bitis)
    df.columns=("Açılış","En Yüksek","En Düşük","Kapanış","Hacim","Temettüler","Bölünmüş")
    st.dataframe(df)
    st.write("Veri Grafiği")
    df = data.history(period="Id", start=baslangic, end=bitis)
    st.line_chart(df["Close"])

    if prophet:

        st.write("Machine Learning Tahmin")
        st.write("(Tarih aralığını değiştirerek tahmininizi geliştirebilirsiniz.)")
        fb=df.reset_index()
        fb['Date'] = fb['Date'].apply(lambda x: x.replace(tzinfo=None))
        fb=fb[['Date', 'Close']]
        fb.columns = ["ds","y"]
        global model
        model=Prophet()
        model.fit(fb)
        future=model.make_future_dataframe(periods=fbperiyot)
        predict=model.predict(future)
        grap=model.plot(predict)
        #predict=predict[["ds","trend"]]
        #predict=predict.set_index("ds")
        #st.line_chart(predict["trend"])
        #plt.show()
        st.write(grap)
        if components:
            grap2=model.plot_components(predict)
            st.write(grap2)
        cvsec = st.sidebar.checkbox("CV")

        if cvsec:
            try:
                st.sidebar.write("#### Metrik Seçiniz")
                metric = st.sidebar.radio("Metrik", ["rmse", "mse", "mape", "mdape"])
                st.sidebar.write("#### Parametre Seçiniz")
                inaralik = range(1, 365)
                cvin = st.sidebar.select_slider("Initial", options=inaralik, value=120)
                initial = str(cvin) + " days"
                peraralik = range(1, 365)
                cvper = st.sidebar.select_slider("CV Periyot", options=peraralik, value=30)
                period=str(cvper) + " days"
                horaralik = range(1, 365)
                cvhor = st.sidebar.select_slider("Horizon", options=horaralik, value=60)
                horizon=str(cvhor) + " days"
                cv = cross_validation(model, initial=initial, period=period, horizon =horizon)
                grap3 = plot_cross_validation_metric(cv, metric=metric)
                st.write(grap3)
            except ValueError:
                st.write("Hata grafiğini görmek için zaman aralığını en az 365 gün girin!")

    else:
        pass


grafikgetir(sembol,baslangic,bitis)



def indir(df):
    csv=df.to_csv()
    b64=base64.b64encode(csv.encode()).decode()
    href=f'<a href="data:file/csv;base64,{b64}">Veri Setini İndir (CSV)</a>'
    return href
st.markdown(indir(df),unsafe_allow_html=True)
def SMA(data,period=30,column="Close"):
    return data[column].rolling(window=period).mean()

def EMA(data,period=21,column="Close"):
    return data[column].ewm(span=period,adjust=False).mean()

def MACD(data,period_long=26,period_short=12,period_signal=9,column="Close"):
    ShortEMA=EMA(data,period_short,column=column)
    LongEMA = EMA(data, period_long, column=column)
    data["MACD"]=ShortEMA-LongEMA
    data["Signal_Line"]=EMA(data,period_signal,column="MACD")
    return data
def RSI(data,period=14,column="Close"):
    delta=data[column].diff(1)
    delta=delta[1:]
    up=delta.copy()
    down=delta.copy()
    up[up<0]=0
    down[down>0]=0
    data["up"]=up
    data["down"] = down
    AVG_Gain=SMA(data,period,column="up")
    AVG_Loss=abs(SMA(data,period,column="down"))
    RS=AVG_Gain/AVG_Loss
    RSI=100.0-(100.0/(1.0+RS))
    data["RSI"]=RSI
    return data

st.sidebar.write("### Finansal İndikatörler")
fi=st.sidebar.checkbox("Finansal İndikatörler")
def filer():
    if fi:
        fimacd = st.sidebar.checkbox("MACD")
        firsi = st.sidebar.checkbox("RSI")
        fisl = st.sidebar.checkbox("Signal Line")
        if fimacd:
            macd=MACD(df)
            st.line_chart(macd["MACD"])
        if firsi:
            rsi = RSI(df)
            st.line_chart(rsi["RSI"])
        if fisl:
            macd = MACD(df)
            st.line_chart(macd["Signal_Line"])

filer()




