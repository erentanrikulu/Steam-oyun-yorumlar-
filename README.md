# Steam-oyun-yorumlar-
# Steam Oyun Yorumları Analizi: Metin Benzerliği Hesaplama ve Değerlendirme

Bu proje, **Steam oyun platformundan alınmış kullanıcı yorumları** üzerinde Doğal Dil İşleme (NLP) teknikleri uygulayarak metinler arası anlamsal benzerliği tespit etmeyi ve farklı metin temsil modellerinin (TF-IDF ve Word2Vec) performansını karşılaştırmayı amaçlamaktadır. Proje kapsamında veri ön işleme, model eğitimi, benzerlik hesaplamaları ve anlamsal kümelendirme analizleri detaylı bir şekilde gerçekleştirilmiştir.

## İçindekiler

1.  [Proje Nedir?](#1-proje-nedir)
2.  [Proje Ne İşe Yarar?](#2-proje-ne-işe-yarar)
3.  [Veri Seti](#3-veri-seti)
4.  [Kurulum](#4-kurulum)
5.  [Proje Yapısı](#5-proje-yapısı)
6.  [Çalıştırma](#6-çalıştırma)
7.  [Kullanılan Modeller ve Yöntemler](#7-kullanılan-modeller-ve-yöntemler)
    * [Veri Ön İşleme](#veri-ön-işleme)
    * [TF-IDF](#tf-idf)
    * [Word2Vec](#word2vec)
    * [Zipf Yasası Analizi](#zipf-yasası-analizi)
    * [Benzerlik Hesaplaması ve Gruplandırma](#benzerlik-hesaplaması-ve-gruplandırma)
8.  [Sonuçlar ve Değerlendirme](#8-sonuçlar-ve-değerlendirme)
    * [Model Karşılaştırması](#model-karşılaştırması)
    * [Model Yapılandırmalarının Etkisi](#model-yapılandırmalarının-etkisi)

---

## 1. Proje Nedir?

Bu proje, **Steam oyun yorumları** gibi büyük ve yapılandırılmamış metin verilerini anlamlandırmak için tasarlanmış bir **Doğal Dil İşleme (NLP) uygulamasıdır.** Amacımız, oyuncuların oyunlar hakkındaki düşüncelerini ifade ettikleri bu yorumları bilgisayarın anlayabileceği sayısal formata (vektörlere) dönüştürmek ve bu vektörler üzerinden yorumlar arasındaki **anlamsal benzerliği** hesaplamaktır.

Proje, bu benzerlik analizi için iki ana metin temsil modelini (TF-IDF ve Word2Vec) kullanır ve bunların oyun yorumları bağlamındaki başarımlarını karşılaştırır. Ayrıca, yorumları anlamsal olarak benzer gruplara ayırmak için kümeleme tekniklerini de uygular. Bu sayede, binlerce yorum arasından ortak temaları veya benzer fikirleri otomatik olarak tespit etme yeteneği kazanılır.

## 2. Proje Ne İşe Yarar?

Bu proje, **Steam oyun yorumları üzerinden elde edilen içgörülerin** çeşitli alanlarda nasıl kullanılabileceğini pratik olarak göstermektedir:

* **Oyun Geliştiricileri İçin Geri Bildirim Analizi:** Oyuncuların yeni özellikler, hatalar veya genel deneyim hakkındaki yorumlarını hızla analiz ederek, hangi yorumların benzer konuları ele aldığını belirlemek. Bu sayede geliştiriciler, oyuncu geri bildirimlerini daha etkin bir şekilde sınıflandırabilir ve ürün iyileştirmeleri için önceliklendirme yapabilir.
* **Pazarlama ve Topluluk Yönetimi:** Belirli bir oyun hakkında çıkan olumlu/olumsuz yorumları veya sıkça dile getirilen temaları otomatik olarak tespit etmek. Bu bilgi, pazarlama kampanyalarını şekillendirmek veya topluluk sorunlarına proaktif çözümler üretmek için kullanılabilir.
* **Oyuncu Tavsiye Sistemleri:** Bir oyuncunun beğendiği veya yaptığı yoruma anlamsal olarak benzer diğer oyun yorumlarını veya potansiyel yeni oyunları önermek. Bu, oyuncu deneyimini kişiselleştirmeye yardımcı olur.
* **Trend Analizi:** Oyun endüstrisindeki belirli dönemlerde popüler olan konuları veya oyuncu beklentilerini yorumlar üzerinden otomatik olarak keşfetmek.
* **Otomatik İçerik Gruplandırma:** Büyük hacimli yorum verilerini (örneğin, forum gönderileri, incelemeler) anlamsal olarak ilişkili kümelere ayırarak veri keşfini kolaylaştırmak.

Özetle, bu proje **Steam oyun yorumları** gibi spesifik bir metin veri setinden anlamlı ve eyleme dönüştürülebilir içgörüler elde etmenin bir yolunu sunar. Aynı zamanda, günümüzdeki büyük dil modellerinin (LLM'ler) temelinde yatan prensipleri küçük ölçekli, gerçek dünya verisi üzerinde deneyimleme fırsatı sağlar.

## 3. Veri Seti

Projemizde, **Steam oyun platformundan alınmış kullanıcı yorumları**ndan oluşan bir veri seti kullanılmıştır.

* **Orijinal Veri Seti:** Başlangıçta 20.000 adet oyun yorumu içeren `20k_veri.csv` dosyası.
* **Kullanılan Alt Küme:** Analiz ve model eğitimi süreçlerinin daha verimli ilerlemesi için, orijinal veri setinden rastgele seçilen **5.000 satırlık** bir alt küme (`veri_5k.csv`) oluşturulmuştur. Bu küçültme işlemi, hesaplama maliyetlerini düşürürken, veri setinin genel özelliklerini korumayı amaçlamıştır.

## 4. Kurulum

Bu projeyi yerel ortamınızda çalıştırmak ve analizleri tekrarlamak için aşağıdaki adımları izleyin:

1.  **Depoyu Klonlayın:**
    ```bash
    git clone [https://github.com/KULLANICI_ADIN/REPO_ADIN.git](https://github.com/KULLANICI_ADIN/REPO_ADIN.git)
    cd REPO_ADIN
    ```
    *(`KULLANICI_ADIN` ve `REPO_ADIN` kısımlarını kendi GitHub kullanıcı adınız ve depo adınız ile değiştirmeyi unutmayın.)*

2.  **Gerekli Kütüphaneleri Yükleyin:**
    Proje için gerekli tüm Python kütüphaneleri aşağıdadır. Bu kütüphaneleri `pip` ile yükleyebilirsiniz:
    ```bash
    pip install pandas numpy scikit-learn nltk gensim matplotlib
    ```
    *Daha iyi bir yaklaşım için, projenin kök dizininde bir `requirements.txt` dosyası oluşturup tüm bağımlılıkları oraya ekleyebilir ve `pip install -r requirements.txt` komutunu kullanabilirsiniz.*

3.  **NLTK Verilerini İndirin:**
    NLTK kütüphanesinin bazı dil kaynaklarına (tokenizer, durak kelimeler, wordnet) ihtiyaç duyulmaktadır. Bu indirmeler, ilgili Jupyter Notebook'lar içindeki ilk kod hücreleri tarafından otomatik olarak yapılacaktır. Ancak, isterseniz manuel olarak da çalıştırabilirsiniz:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

## 5. Proje Yapısı

Proje, Steam oyun yorumları üzerinde gerçekleştirilen veri işleme ve analiz adımlarını takip eden, mantıksal olarak düzenlenmiş bir dizi Jupyter Notebook (`.ipynb`) dosyasından oluşmaktadır. Her bir notebook, belirli bir aşamayı temsil eder:

* `5k-düsürme.ipynb`: Orijinal `20k_veri.csv` dosyasından 5.000 adet rastgele Steam yorumu seçerek `veri_5k.csv` adlı yeni bir dosya oluşturur. Bu, daha verimli bir çalışma ortamı sağlar.
* `dogal-dil.ipynb`: `veri_5k.csv` içindeki Steam yorumları üzerinde çeşitli doğal dil ön işleme adımlarını uygular: küçük harfe çevirme, tokenizasyon (kelimelere ayırma), noktalama işaretlerini ve sayıları temizleme, durak kelimeleri kaldırma, kök bulma (stemming) ve lemmatizasyon. Bu adım, yorumların modellere hazır hale gelmesini sağlar.
* `tfidf-lemmatized.ipynb`: Lemmatize edilmiş Steam yorumları metinleri üzerinde TF-IDF (Term Frequency-Inverse Document Frequency) modelini eğitir ve elde edilen vektörleri `tfidf_lemmatized.csv` olarak kaydeder. Ayrıca, "game" kelimesi gibi örnek bir kelime için benzer kelimeleri gösterir.
* `tfidf-stemmed.ipynb`: Kök bulma (stemming) işlemi uygulanmış Steam yorumları metinleri üzerinde TF-IDF modelini eğitir ve elde edilen vektörleri `tfidf_stemmed.csv` olarak kaydeder. "game" kelimesi için benzer kelime örnekleri sunar.
* `word2vec.ipynb`: Hem lemmatize edilmiş hem de stemmed edilmiş Steam yorumları metin listeleri üzerinde çeşitli parametrelerle (CBOW/Skip-Gram, pencere boyutu: 2/4, vektör boyutu: 100/300) toplam 8 farklı Word2Vec modelini eğitir ve her birini `.model` uzantısıyla kaydeder. Oyun yorumlarındaki en sık kullanılan kelimeleri de listeler.
* `zipf-analiz.ipynb`: Ham (ön işleme yapılmamış) Steam yorumları metinleri üzerinde Zipf Yasası'nı analiz eder ve kelime frekans dağılımını görselleştirir. Bu, veri setinin dilsel özelliklerini anlamaya yardımcı olur.
* `zipf-lemmatized.ipynb`: Lemmatizasyon uygulanmış Steam yorumları metinleri üzerinde Zipf Yasası analizi yapar.
* `zipf-stemmed.ipynb`: Kök bulma (stemming) uygulanmış Steam yorumları metinleri üzerinde Zipf Yasası analizi yapar.
* `metin-benzerlik.ipynb`: Eğitilmiş tüm TF-IDF ve Word2Vec modellerini yükleyerek, **belirlenen bir Steam yorumuna (örneğin, veri setindeki ilk yorum)** en benzer ilk 5 diğer yorumu bulur. Ayrıca, farklı modellerin bulduğu en benzer yorum listeleri arasındaki Jaccard Benzerliğini hesaplayarak modellerin karşılaştırmalı performansını değerlendirir. Bu notebook, projenin temel sonuçlarını sunar.
* `cosine-benzerlik-grup.ipynb`: Steam yorumları üzerinde Cosine Benzerliği ve Hiyerarşik Kümeleme (`AgglomerativeClustering`) kullanarak anlamsal olarak benzer yorumları gruplara ayırır. Her küme için en sık geçen kelimeye göre bir başlık oluşturur ve grupları listeler.
* `20k_veri.csv`: Projede kullanılan orijinal, büyük Steam oyun yorumları veri seti. **(Not: Bu dosya büyük olabileceğinden, depoya dahil edilmeyebilir veya ayrı bir depolama çözümü gerekebilir.)**
* `veri_5k.csv`: `5k-düsürme.ipynb` tarafından oluşturulan, 5.000 satıra düşürülmüş Steam oyun yorumları veri seti.

## 6. Çalıştırma

Projedeki tüm analizleri baştan sona çalıştırmak ve sonuçları yeniden üretmek için Jupyter Notebook dosyalarını **sırasıyla ve belirtilen adımları takip ederek** çalıştırmanız önemlidir:

1.  **Veri Setini Hazırlama:**
    * `5k-düsürme.ipynb` dosyasını çalıştırın. Bu adım, tüm analizlerin temelini oluşturan `veri_5k.csv` dosyasını oluşturacaktır.

2.  **Steam Yorumları Ön İşleme:**
    * `dogal-dil.ipynb` dosyasını çalıştırın. Bu notebook, `veri_5k.csv` içindeki yorumlar üzerinde gerekli tüm dilsel ön işleme adımlarını uygulayarak, modellerin kullanabileceği temizlenmiş metin listelerini hazırlar.

3.  **Metin Temsil Modellerini Eğitme:**
    * `tfidf-lemmatized.ipynb` dosyasını çalıştırın.
    * `tfidf-stemmed.ipynb` dosyasını çalıştırın.
    * `word2vec.ipynb` dosyasını çalıştırın. Bu adımlar, Steam yorumlarının sayısal vektör temsillerini öğrenen TF-IDF ve Word2Vec modellerini oluşturacak ve kaydedecektir.

4.  **Zipf Yasası Analizi (Kelime Dağılımını Keşfetmek İçin):**
    * `zipf-analiz.ipynb`
    * `zipf-lemmatized.ipynb`
    * `zipf-stemmed.ipynb`
    * Bu notebook'ları çalıştırarak Steam yorumlarındaki kelime frekans dağılımlarını ve doğal dil özelliklerine uygunluklarını görsel olarak inceleyebilirsiniz.

5.  **Metin Benzerliği Hesaplama ve Model Karşılaştırması (Projenin Ana Çıktısı):**
    * `metin-benzerlik.ipynb` dosyasını çalıştırın. Bu notebook, eğitilmiş tüm modelleri kullanarak Steam yorumları arasındaki anlamsal benzerlikleri hesaplar ve modellerin performansını karşılaştıran ana sonuçları sunar.

6.  **Steam Yorumlarını Gruplandırma (İsteğe Bağlı Analiz):**
    * `cosine-benzerlik-grup.ipynb` dosyasını çalıştırın. Bu notebook, benzer Steam yorumlarını otomatik olarak kümelere ayırarak, yorum verilerindeki ortak temaları keşfetmenize yardımcı olur.

Her notebook dosyasının içindeki kod hücrelerini sırayla çalıştırmayı ve çıktıları gözlemlemeyi unutmayın.

## 7. Kullanılan Modeller ve Yöntemler

Bu proje, Steam oyun yorumları üzerinde anlamsal benzerlik ve kümelendirme analizleri yapmak için çeşitli NLP tekniklerini ve makine öğrenimi modellerini kullanmıştır.

### Veri Ön İşleme (Steam Yorumları İçin)

Steam yorumları, ham halleriyle analiz için uygun değildir. Bu nedenle, metinlerin temizlenmesi ve standartlaştırılması için aşağıdaki adımlar uygulanmıştır:

* **Küçük Harfe Çevirme:** Tüm yorumlar küçük harfe dönüştürülerek "Game" ve "game" gibi kelimelerin aynı kabul edilmesi sağlanmıştır.
* **Tokenizasyon:** Yorumlar, tek tek kelimelere veya cümlelere ayrılmıştır (`nltk.word_tokenize`, `nltk.sent_tokenize`).
* **Noktalama ve Sayı Temizleme:** Metinlerdeki noktalama işaretleri, özel karakterler ve sayılar, yorumların saf kelime içeriğini korumak için kaldırılmıştır.
* **Durak Kelime (Stop Word) Temizliği:** "a", "the", "is" gibi metnin anlamsal içeriğine çok az katkı sağlayan, ancak sıklıkla geçen kelimeler (stop words), NLTK'nin İngilizce stop word listesi kullanılarak yorumlardan çıkarılmıştır.
* **Kök Bulma (Stemming - Porter Stemmer):** Kelimeler, anlamlarını korumayabilir, ancak aynı kökten gelen farklı çekimlerin (örn: "running", "runs", "ran" -> "run") tek bir formda temsil edilmesini sağlamak için köklerine indirgenmiştir.
* **Lemmatizasyon (WordNet Lemmatizer):** Kelimeler, anlamlarını koruyarak sözlükteki temel hallerine (lemma) indirgenmiştir (örn: "better" -> "good"). Stemming'e göre daha gelişmiş bir yöntem olup, anlamsal bütünlüğü daha iyi korur.

### TF-IDF (Steam Yorumlarının Anahtar Terim Ağırlıkları)

* **Tanım:** TF-IDF (Term Frequency-Inverse Document Frequency), bir kelimenin belirli bir Steam yorumu içindeki önemini hem o yorumdaki sıklığına (Term Frequency) hem de tüm yorum veri setindeki genel sıklığına (Inverse Document Frequency) göre ölçen istatistiksel bir yöntemdir. Nadir ve spesifik kelimeler genellikle daha yüksek TF-IDF skorları alır.
* **Uygulama:** `sklearn.feature_extraction.text.TfidfVectorizer` sınıfı kullanılarak ön işlenmiş Steam yorumları, TF-IDF vektörlerine dönüştürülmüştür.
* **Benzerlik Metriği:** TF-IDF vektörleri arasındaki benzerliği ölçmek için **Cosine Benzerliği** kullanılmıştır.

### Word2Vec (Steam Yorumları için Anlamsal Gömülmeler)

* **Tanım:** Word2Vec, kelimelerin anlamsal anlamlarını çok boyutlu bir vektör uzayında temsil eden bir kelime gömme (word embedding) modelidir. Kelimeler, Steam yorumlarındaki bağlamlarına göre öğrenilir ve anlamsal olarak benzer kelimeler vektör uzayında birbirine yakın konumlandırılır. Bu sayede, "oyun" ve "eğlence" gibi kelimelerin yakın ilişkileri yakalanabilir.
* **Uygulama:** `gensim.models.Word2Vec` kütüphanesi kullanılarak Steam yorumları üzerinde çeşitli Word2Vec modelleri eğitilmiştir.
* **Parametreler:** Farklı model yapılandırmalarının etkisini incelemek için şunlar denenmiştir:
    * `model_type`: CBOW (Continuous Bag-of-Words) ve Skip-Gram.
    * `window`: 2 ve 4 (hedef kelimenin etrafındaki bağlam kelimelerinin sayısı).
    * `vector_size`: 100 ve 300 (her kelimenin temsil edileceği vektörün boyutu).
* **Benzerlik Metriği:** Word2Vec vektörleri arasındaki anlamsal benzerliği ölçmek için **Cosine Benzerliği** kullanılmıştır.

### Zipf Yasası Analizi (Steam Yorumlarının Dilsel Yapısı)

* **Tanım:** Zipf Yasası, doğal dillerde kelime frekanslarının belirli bir dağılım sergilediğini belirtir: en sık kullanılan kelimenin frekansının yaklaşık yarısı kadar sıklıkla ikinci en sık kullanılan kelimeye rastlanır ve bu böyle devam eder.
* **Uygulama:** Ham, lemmatize edilmiş ve stemmed edilmiş Steam yorumları veri setleri üzerinde Zipf Yasası'na uygunluk analizleri yapılarak kelime frekans dağılımları görselleştirilmiştir. Bu, veri setinin doğal dil özelliklerine uygun olup olmadığını ve ön işleme adımlarının kelime dağılımını nasıl etkilediğini anlamaya yardımcı olmuştur.

### Benzerlik Hesaplaması ve Gruplandırma (Steam Yorumları Üzerine)

* **Metinler Arası Benzerlik (`metin-benzerlik.ipynb`):** Eğitilen TF-IDF ve Word2Vec modelleri kullanılarak, belirli bir Steam yorumuna en benzer diğer yorumlar bulunmuştur. **Cosine Benzerliği**, iki yorum vektörü arasındaki yönsel yakınlığı (anlamsal benzerliği) ölçmek için kullanılmıştır.
* **Model Sonuçlarının Karşılaştırılması:** Farklı modellerin bulduğu "en benzer ilk 5 Steam yorumu" listeleri arasındaki ortaklık, **Jaccard Benzerliği** (iki küme arasındaki kesişim kümesinin birleşim kümesine oranı) kullanılarak değerlendirilmiştir. Bu, farklı modellerin ne kadar benzer sonuç kümeleri ürettiğini nicel olarak anlamamızı sağlamıştır.
* **Metin Gruplandırma (`cosine-benzerlik-grup.ipynb`):** Özellikle TF-IDF vektörleri kullanılarak Steam yorumları arasında anlamsal kümelendirme (clustering) yapılmıştır. `AgglomerativeClustering` (Hiyerarşik Kümeleme) algoritması ile anlamsal olarak benzer yorumlar otomatik olarak gruplara ayrılmış ve her grup için içeriği özetleyen en sık geçen kelime bir başlık olarak belirlenmiştir. Bu, büyük yorum veri setlerinde ortak temaları keşfetmek için güçlü bir araçtır.

## 8. Sonuçlar ve Değerlendirme

Bu bölümde, Steam oyun yorumları üzerinde eğitilen modellerin metin benzerliği hesaplamalarındaki başarımları ayrıntılı olarak incelenmekte ve karşılaştırılmaktadır.

### Model Karşılaştırması

Yapılan analizlerde (özellikle `metin-benzerlik.ipynb` çıktısı baz alınarak):

* **Word2Vec modelleri, TF-IDF modellerine kıyasla Steam yorumları arasındaki anlamsal benzerliği yakalamada önemli ölçüde daha başarılı olmuştur.** Word2Vec tarafından elde edilen Cosine benzerlik skorları (genellikle 0.94-0.97 aralığında) TF-IDF'e (genellikle 0.23-0.28 aralığında) göre belirgin şekilde yüksektir. Bu durum, Word2Vec'in kelimelerin bağlam içindeki ilişkilerini öğrenme yeteneğinin, oyun yorumları gibi anlamsal derinlik içeren metinlerde TF-IDF'e göre daha üstün olduğunu göstermektedir.
* **TF-IDF ve Word2Vec, Steam yorumları için anlamsal benzerliği çok farklı şekillerde yorumlamıştır.** `metin-benzerlik.ipynb` çıktısındaki Jaccard Benzerlik matrisinde, TF-IDF modelleri ile Word2Vec modelleri arasındaki Jaccard benzerliği $0.00$ olarak gözlemlenmiştir. Bu, iki model ailesinin aynı girdi yorumu için tamamen farklı "en benzer" yorumları bulduğunu ve metin benzerliğine yaklaşımlarının temelden farklı olduğunu net bir şekilde ortaya koymaktadır. TF-IDF, kelime sıklığına dayalı "bag-of-words" yaklaşımıyla daha yüzeysel benzerlikleri yakalarken, Word2Vec anlamsal ilişkiler üzerinden daha derinlemesine benzerlikleri ortaya çıkarmıştır.

### Model Yapılandırmalarının Etkisi

`metin-benzerlik.ipynb` çıktısındaki Jaccard matrisi ve gözlemlenen benzerlik skorları, Word2Vec modellerinin Steam yorumları üzerinde sergilediği tutarlılığı ortaya koymaktadır:

* **CBOW vs. Skip-Gram:** Her iki Word2Vec mimarisi de aynı en benzer yorumları bulmuştur. Bu, bu özel Steam yorumları veri seti ve benzerlik görevi için CBOW ve Skip-Gram arasında belirgin bir performans farkı olmadığını göstermektedir.
* **Pencere Boyutu (Window Size):** 2 ve 4 gibi farklı pencere boyutları denenmesine rağmen, benzerlik sonuçlarında bir değişiklik gözlemlenmemiştir. Bu, Steam yorumlarındaki anlamsal yakınlığı yakalamak için daha küçük bir bağlam penceresinin bile yeterli olduğunu düşündürmektedir.
* **Vektör Boyutu (Vector Size):** 100 ve 300 gibi farklı vektör boyutları kullanılmasına rağmen, sonuçlar üzerinde anlamlı bir farklılık oluşmamıştır. Bu durum, 100 boyutlu vektörlerin bile bu veri setindeki anlamsal ilişkileri başarılı bir şekilde temsil edebildiğini göstermektedir. Daha yüksek boyutlar, daha karmaşık anlamsal özellikleri yakalama potansiyeli sunsa da, bu senaryoda performans artışı sağlamamıştır.
* **Ön İşleme (Lemmatized vs. Stemmed):** Word2Vec modelleri için hem lemmatize edilmiş hem de stemmed edilmiş Steam yorumları kullanılmış ve yine aynı en benzer yorumlar bulunmuştur. Bu, Word2Vec modelinin kendisinin, ön işlemedeki bu ince farklara karşı oldukça dirençli olduğunu veya öğrenilen anlamsal ilişkilerin bu seviyede bir farklılıktan etkilenmediğini göstermektedir.

Genel olarak, Word2Vec modellerinin Steam yorumları üzerindeki performansının, parametre seçiminden (belirli bir aralıkta) çok, modelin anlamsal kelime gömme yeteneğinin TF-IDF'e kıyasla üstün olmasından kaynaklandığı söylenebilir.
