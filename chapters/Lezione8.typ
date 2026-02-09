#import "../template.typ": *
= Intro Deep Learning

== Funzioni di Attivazione

I modelli di deep learning sono in grado di catturare informazioni *al di là dell'osservabile*, grazie all'uso delle funzioni di attivazione che introducono non-linearità nel modello.

=== Caratteristiche delle Funzioni di Attivazione

Le *funzioni di attivazione* presentano caratteristiche specifiche:
- Hanno tipicamente *due asintoti* (da $+infinity$ a $-infinity$)
- Per ogni valore $x$ tendono ad essere in *saturazione* tra $-1$ e $1$ (o $0$ e $1$ a seconda della funzione)
- La *parte centrale* rappresenta la regione di incertezza del modello

#nota()[
  Le funzioni di attivazione servono per definire *regioni non lineari* nello spazio, creando curvature che permettono di separare gruppi di dati in base alla loro categoria di appartenenza. In uno stesso spazio di embedding, creano superfici di separazione complesse.
]

#esempio()[
  *Tangente iperbolica* ($tanh$): Una delle funzioni di attivazione classiche, definita come:
  $
    tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
  $
  Ha output nel range $[-1, 1]$ con saturazione agli estremi e maggiore sensibilità al centro.
]

=== ReLU (Rectified Linear Unit)

La *ReLU* è una delle funzioni di attivazione più utilizzate nel deep learning moderno. È definita come:
$
  "ReLU"(x) = max(0, x)
$

#nota()[
  La ReLU restituisce $0$ quando $x < 0$ e $x$ quando $x >= 0$. Applicando una trasformazione lineare si ottiene: $R(w x + b)$ che dipende dal valore della retta $w x + b$.
]

==== Teorema di Approssimazione Universale con ReLU

Data una funzione continua $f in C([a,b], RR)$, è possibile *approssimala come combinazione lineare di ReLU*.

Il principio di funzionamento è il seguente:
+ Prendiamo tante ReLU con diverse traslazioni e scalature
+ Combinandole opportunamente, otteniamo una *curva spezzata* che approssima $f$
+ Più ReLU utilizziamo, migliore è l'approssimazione (ma aumenta la complessità)

#esempio()[
  Combinando una funzione ReLU "rossa" e una "blu" con opportuni pesi, possiamo creare una funzione risultante che si avvicina alla funzione target (grigia) che vogliamo approssimare.
]

#attenzione()[
  *Limitazioni*:
  - Il modello può diventare *molto complesso* con molte unità
  - Richiede *più tempo di training* e *maggiore quantità di dati* per raggiungere un'accuratezza accettabile
  - Trade-off tra capacità di approssimazione e costo computazionale
]

==== Generalizzazione del Teorema

Il teorema può essere generalizzato: per ogni funzione continua $g:[0,1]^D -> RR$ è possibile approssimarla con *un singolo livello nascosto* (percettrone) utilizzando:
- $W in RR^(K times D)$ (matrice dei pesi)
- $b in RR^K$ (vettore di bias)
- $omega in RR^K$ (pesi di output)

#nota()[
  Aumentando $K$ (ovvero il numero di *unità nascoste*), abbiamo più possibilità di apprendere funzioni complesse, ma con un costo computazionale maggiore.
]

=== Modelli Profondi vs Modelli Flat

Il vantaggio principale dei *modelli profondi* (deep) rispetto ai modelli superficiali (flat) è che le *reti neurali gerarchiche* con diversi livelli di profondità sono in grado di estrarre *più informazioni strutturate* dai dati.

==== Rappresentazioni Gerarchiche

I modelli profondi sfruttano *bias induttivi* (scelti dal progettista) per apprendere pattern sempre più complessi attraverso i livelli:

#esempio()[
  *Visione artificiale*:
  - *Primi livelli*: catturano pattern basilari come edge, angoli, texture
  - *Livelli intermedi*: combinano i pattern basilari in strutture più complesse (parti di oggetti)
  - *Ultimi livelli*: comprendono l'immagine nel suo complesso, riconoscendo oggetti e scene
]

#nota()[
  Il *learning* è una stima di parametri sempre più efficaci, utilizzando bias induttivi (embedding e livelli intermedi) per favorire l'apprendimento di *rappresentazioni gerarchiche* dei dati.
]

== PyTorch nn Framework

Utilizziamo il modulo `torch.nn`, essenziale per costruire modelli di deep learning in PyTorch.

=== Componenti Principali

Il framework `torch.nn` fornisce diversi moduli fondamentali:

/ Layer: Tutti i vari *livelli* della rete. Esistono layer ricorrenti, lineari, convoluzionali, ecc.
/ Funzioni di attivazione: ReLU, Sigmoid, Tanh, Softmax, ecc.
/ Livelli di normalizzazione: BatchNorm, LayerNorm, ecc.
/ Funzioni di perdita (loss): MSE, CrossEntropy, BCE, ecc.

=== Costruzione di un Modello Custom

Quando creiamo un nuovo modello, dobbiamo costruire una *sotto-classe di `nn.Module`*. La struttura base è:

```python
class ModelNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Definizione dei layer

    def forward(self, x):
        # Definizione del flusso dei dati
        return output

```
=== Layer Lineari (Trasformazioni Affini)

Una *trasformazione lineare* in PyTorch è implementata come una *trasformazione affine*:
$
  y = A overline(x) + overline(b)
$

dove $A$ è una matrice e $overline(b)$ è il vettore di bias.

==== Rappresentazione Matriciale

Moltiplicando le righe di $A$ per il vettore colonna $overline(x)$ e sommando il bias, otteniamo l'output.

Se applichiamo la trasposizione a entrambi i membri:
$
  y & = (A overline(x))^t + overline(b)^t \
  y & = overline(x)^t A^t + overline(b)^t
$

Equivalentemente (e più efficientemente per la rappresentazione in PyTorch):
$
  y = overline(x) W^t + overline(b)
$

#nota()[
  *Dimensioni dei tensori*:
  - *Input*: `(batch_size, in_features)`
  - *Output*: `(batch_size, out_features)`

  Le features differiscono in base alle dimensioni della matrice $W in RR^("out_features" times "in_features")$ e del vettore bias $b in RR^("out_features")$.
]

#esempio()[
  Creazione e utilizzo di un layer lineare:
  ```python
  layer = nn.Linear(in_features=5, out_features=3)
  x = torch.randn(2, 5)  # batch_size = 2
  y = layer(x)
  print(y.shape)  # torch.Size([2, 3])
  ```

  L'importante è che il tensore corrisponda sulle `in_features` (dimensione 5). Il risultato avrà dimensione determinata da `out_features` (dimensione 3).
]

==== Composizione Sequenziale di Layer

Un uso tipico prevede la composizione di più layer in sequenza:

```python
model = nn.Sequential(
    nn.Linear(784, 128),  # Input layer
    nn.ReLU(),            # Activation
    nn.Linear(128, 10)    # Output layer
)
```

#nota()[
  I layer devono essere *compatibili*: l'output di un layer deve corrispondere all'input del layer successivo. Nell'esempio: $784 -> 128 -> 128 -> 10$.
]

Quando passiamo un input `x` al modello, vengono applicati in sequenza tutti i layer definiti
$
  y & = (A overline(x))^t+overline(b)^t \
  y & = overline(x)^t A^t + overline(b)^t
$
Equivale a scrivere (usanPesi

All'inizio del training, la *matrice dei pesi* (che dovrà essere appresa) viene *inizializzata casualmente*.

#attenzione()[
  L'inizializzazione dei parametri è fondamentale per il successo del training. Esiste un'intera branca di ricerca che studia come inizializzare i parametri: *da dove si parte influenza fortemente come il modello evolve* durante l'addestramento.
]

#esempio()[
  *Xavier Normad Error (MSE) Loss*

  Calcola la *differenza quadratica* tra valori predetti e valori reali, elemento per elemento:
  $
    L_"MSE" = 1/N sum_(i=1)^N (y_i - hat(y)_i)^2
  $
]

*Proprietà*:
- Penalizza fortemente quando gli elementi sono molto differenti tra loro (errori grandi pesano quadraticamente)
- Sensibile agli outlier
- Usata principalmente in problemi di *regressione*

=== L1 Loss (Mean Absolute Error)

Misura la *Cross-Entropy (BCE)*

La *Binary Cross-Entropy* è utilizzata per problemi di *classificazione binaria*:
$
  L_"BCE" = -1/N sum_(i=1)^N [y_i log(hat(y)_i) + (1-y_i) log(1-hat(y)_i)]
$

==== Concetto di Entropia

L'*entropia* è una misura informazionale che quantifica il "tasso di sorpresa" o l'incertezza di una distribuzione di probabilità.

#esempio()[Multiclasse]

Nella *classificazione multiclasse* (più di 2 classi), il modello produce diversi valori (chiamati *logits*) in uscita, uno per ogni classe. Per convertirli in probabilità e calcolare la loss, usiamo *Softmax* e *Negative Log-Likelihood*.

=== Softmax

La funzione *Softmax* converte i logits in una distribuzione di probabilità:
$
  "softmax"(z_i) = (e^(z_i))/(sum_(j=1)^K e^(z_j))
$

dove $K$ è il numero di classi.

#nota()[
  La Softmax garantisce che:
  - Ogni output sia $in [0,1]$
  - La somma di tutte le probabilità sia uguale a $1$
  - Ogni valore rappresenta $P(y = k | x)$, la probabilità che $x$ appartenga alla classe $k$
]

=== Verosimiglianza e Log-Likelihood

Dato un dataset $D = {(x_i, y_i)}_(i=1)^N$ con label $y_i$ e input $x_i$, consideriamo un modello $hat(y)_i = M_theta(x_i)$.

==== Caso Binario (Richiamo)

Per un problema binario $y_i in {0, 1}$ con modello $theta in [0,1]$, la *verosimiglianza* (likelihood) è:
$
  P(D | theta) = product_(i=1)^N hat(y)_i^(y_i) (1-hat(y)_i)^(1-y_i)
$

#nota()[
  La probabilità è molto alta quando il modello predice correttamente tutti gli esempi.
]

Usiamo la *log-likelihood* perché:
- La produttoria diventa somma (più stabile numericamente)
- Evitiamo problemi di underflow con $N$ grande
- È più facile da ottimizzare

$
  log P(D | theta) = sum_(i=1)^N [y_i log hat(y)_i + (1-y_i) log(1-hat(y)_i)]
$

==== Caso Multiclasse

Per la classificazione multiclasse con $y_i in {1, 2, dots, K}$, la *log-likelihood* diventa:
$
  log P(D | theta) = sum_(i=1)^N log p(y_i | x_i)
$

=== Cross-Entropy Loss per Classificazione Multiclasse

La *Cross-Entropy Loss* è definita come la *Negative Log-Likelihood*:
$
  L_"CE" = -1/N sum_(i=1)^N log p(y_i | x_i) = -1/N sum_(i=1)^N log hat(y)_(i,j)
$

dove $j$ è l'indice della classe corretta per l'esempio $i$-esimo.

==== One-Hot Encoding

Le label $y_i in {1, 2, dots, K}$ vengono rappresentate in *one-hot encoding*:
- Un vettore di dimensione $K$ con tutti $0$ tranne un $1$ in posizione $j$ (classe corretta)
- Esempio: per $K=3$ e classe $2$: $y = [0, 1, 0]$

#esempio()[
  Supponiamo $K = 3$ classi e l'esempio $i$-esimo ha label vera $j = 2$:
  - Il modello produce: $hat(y)_i = (alpha_1, alpha_2, alpha_3)$ dopo softmax
  - La loss considera solo: $log hat(y)_(i,2) = log alpha_2$
  - Idealmente vogliamo $alpha_2 approx 1$ e $alpha_1, alpha_3 approx 0$
]

#nota()[
  Il *one-hot encoding* agisce come una *maschera* che seleziona solo la probabilità predetta per la classe corretta $j$. La log-likelihood tiene conto solo del valore $hat(y)_(i,j)$ dove $j$ è la classe vera dell'esempio $i$-esimo.
]

#attenzione()[
  *Obiettivo*: vogliamo trovare il miglior $theta$ (parametri del modello) che massimizza la log-likelihood (o equivalentemente minimizza la negative log-likelihood). Questo corrisponde a trovare il modello che spiega meglio il dataset $D$, tenendo conto dei bias induttivi che abbiamo introdotto nell'architettura.
]
  nn.linear(784,128)
  nn.ReLu()
  nn.linear(128,10)
  compatibili, output matcha con input


Alla x  vengono applicati in sequena i layer che ho definito.

=== Inizializzazione dei pesi

All'inizio la matrice dei pesi (che dovrà essere imparata) è inizializzata casualmente. Esiste una branca che studia come inizializzare i parametri, da dove si parte dipende poi come il modello evolve.
`nn.init.xavier_normal(layer.weight)` permette di creare una matrice inizializzata secondo una certa distribuzione.

//guardare image classifier

Funzioni di loss.

=== Mean Square loss
Calcola la differenza quadratica elemento ad elemento.
$
  L_"MSE" = 1/N sum_(i=1)^N (y_i-hat(y)_i)^2
$
Pensalizza se gli elementi sono molto differenti tra di loro.

=== L1Loss
Misura la differenza media assoluta. più sensible agli outlier. Usata in regressione.

#nota()[
  Son funzioni entrambe convesse, essenziali per la minimiazzaione.
]

=== Binary cross-entropy
$
  L_"BCE" = - 1/N sum_(i=1)^N [y_i log(hat(y)_i)+(1-y_i)log(1-hat(y)_i)]
$
L'entropia misura il tasso di sopresa. Se prendiamo un enciclopedia qual'è la probabilità di ogni singola lettera dell'alfabeto. Per una sorgente come l'enciclipedia misura come sono distrubite le lettere, quando lego una parola lettera per lettera è la sopresa nel leggere la prossima.
Ci dice quanto una distibuzione di probabilità abbia più o meno caos. Date due distribuzioni di probabilità $P_i$ e $P_j$. Quando ho $P_j = 1/k$ con $k$ numero simboli ho il massimo del caos.

In questo caso ho un classificatore binario $0 "e" 1$. Chidiamo al modello di sputare fuori un valore che sia in [0,1]. Idialmente il modello restituisce il modello restituisce a 0 quando la pool label è molto vicina a 0.

Per ongi elemento del dataset $i$, se la verità è $0$ la prima parte viene cancellata e vale la seconda. Quando la verità è $1$ la seconda parte della formula viene persa. La loss penalizza gli errori in base alla label che mi interessa (concetto entropico).

== Classificazione multi classe

Se facessi classificazione multiclasse e non binaria ho parecchi valori (logits) usciti dal modello. Applico il softmax e applico la negative log-likehood.

Per ogni valore sputa una distribuzione di probabilità ognunga per ogni classe che abbiamo, ognuna rappresenta la probabilità che $x$ appartenga alla classe $k$.

I logits sono dei valori a caso. Per riportarli nelle probabilità uso la funzione softmax

Soft max restitusice una somma di probabilità con somma 1. Se prendo un dataset $D=[(x_i,y_i)]_(i=1)^n$ con label $y_i$ e dato $x_i$. Se prendo un modello $overline(y_i) = theta(M_(theta(x_i)))$ se è binario $y_i = in {0,1}$ allora il modello $theta$ appartiene a [0,1].

la verosimiglianza del modello è data da:
$
  P(D| theta) = product overline(y_i)^(1-overline(y_i))^(1-y_i)
$
La probabità è molto alta quando azzecco tutto. Di solito si usa la log-likehood la produttoria si trasforma in somma (produttoria molto brutta se N è molto grande non sforo rappresentazione e se ho molte predizioni sbagliate è un problema):
$
  log(P(..)) = sum_(i=1)^N y_i log overline(y_i) ...
$
Io voglio il miglior $theta$ che spiega in modo migliroe il dataset $D$ (applicando bias). Otteniamo così il miglir mdoello ovvero il miglior theta per D.

//aggiugnere formula softmax
Nella softmax vado a definire la cross-entropy loss:
$
  L = -1/N ....
$
Adesso ho che $y_i = {1,2,dots,K}$ dove $K$ è il numero di classi. la mia log-likehood diventa:
$
  P(D | theta) = -1/N sum_(k=1)^N log p(y_k | x_k) = -1/N sum_(k=1)^N log(y).
$
i k valori che devo rappresentare li esprimo in one hote encoding (solo un 1 in posizione $j$). facendo $log overline(y)_(i,j)$ selezione solo quello di indice $j$. La maschera seleziona solo la classe $j$ e cstruisco una log-likehood che tiene conto del fatto che per l'esemplare $i$-esimo sto considerando la $j$-esima classe che mi da il valore.\
$overline(y)$ è grande come $y_i$ e $overline(y)_i = (alpha, .. alpha_k)$ è la somma di k valori dove si spera che a spiccare sia quello di posto $j$ ovvero la classe corretta.
