#import "../template.typ": *
= Intro Deep Learning

Possiamo vedere un modello di deep learning come una _scatola nera_ il cui compito è apprendere un set di *parametri $Theta$* (detti anche iper-parametri).\
Dati:
- Osservazioni ${(mr(x_i),mb(y_i))}$, dove:
  - *$mr(x_i)$* sono ciò che ha predetto il modello
  - *$mb(y_i)$* è la _label_ corretta
- Un modello parametrico: $f(x, Theta)$

La fase di apprendimento (learning) consiste nel trovare il set di parametri $theta^*$ tale che:
$
  theta^* = arg min_(Theta) mr(L)(f(x, Theta), y)
$
Dove $mr(L)$ è una *funzione di loss*.

La predizione di una rete neurale ($x$) è quindi data da:
$
  x -> f(x, mr(W), mb(b))\
  Theta = underbrace(mr(W), "matrice"\ "pesi"),underbrace(mb(b), "bias")
$

La fase di *training* consiste nei seguenti passaggi (in loop):
- *forward pass*: vengono computate le predizioni, per un certo batch (sotto-insieme) di dati
- *loss*: le predizioni vengono comparate con la *ground truth*
- *backpropagation*: vengono computati i gradienti
- *update parameters*: vengono aggiornati il set di pesi $W$ del modello, in modo da minimizzare la loss.

#informalmente()[
  Quello che accade è che dato un input $x$, esso subisce varie _trasformazioni_ ovvero passa per diversi *layer nascosti* del modello. Ogni layer nascosto è formato da una fila di neuoroni, ciascuno con la propria funzione di attivazione. La predizione finale si ottiene nel seguente modo:
  $
    mg(o) = mr(tanh)(mb(W_(h))(dots mr(tanh)(mb(w_(2))(mr(tanh)(mb(w_1) mo(x) + mb(b_1))+mb(b_2)))dots +mb(b_n)))
  $
  Dove:
  - $mo(x)$ è l'input
  - $mg(o)$ è la predizione del modello
  - $mr(tanh)$ è la funzione di attivazione
  - $mb(W_h)$ e $mb(b_h)$ sono i _pesi_ ai vari livelli che il modello apprende
]







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

/*
#figura(
  ```python
  import cetz.plot

  cetz.canvas({
    import cetz.draw: *

    cetz.plot.plot(
      size: (10, 6),
      x-label: $x$,
      y-label: $f(x)$,
      x-tick-step: 1,
      y-tick-step: 0.5,
      x-min: -3,
      x-max: 3,
      y-min: -0.5,
      y-max: 3,
      {
        // ReLU
        cetz.plot.add(
          ((x,) => (calc.max(0, x),)),
          domain: (-3, 3),
          samples: 100,
          style: (stroke: (paint: mo, thickness: 2pt)),
          label: [ReLU]
        )

        // Tanh (per confronto)
        cetz.plot.add(
          ((x,) => (calc.tanh(x),)),
          domain: (-3, 3),
          samples: 100,
          style: (stroke: (paint: mb, thickness: 1.5pt, dash: "dashed")),
          label: [Tanh]
        )

        // Sigmoid (per confronto)
        cetz.plot.add(
          ((x,) => (1 / (1 + calc.exp(-x)),)),
          domain: (-3, 3),
          samples: 100,
          style: (stroke: (paint: mr, thickness: 1.5pt, dash: "dotted")),
          label: [Sigmoid]
        )
      }
    )
  })
  ```
  caption: [Confronto tra funzioni di attivazione: ReLU (arancione), Tanh (blu tratteggiato) e Sigmoid (rosso punteggiato)]
)
*/

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
/*
#figura(
  ```python
  import fletcher as fl

  fl.diagram(
    node-stroke: 1pt,
    spacing: (20mm, 8mm),
    edge-stroke: 1.5pt,
    mark-scale: 70%,
    {
      let layers = (
        (pos: (0, 0), label: [Input\n$mb(x)$], fill: mb),
        (pos: (0, 1), label: [Linear\n$W_1 mb(x) + mb(b)_1$], fill: mo),
        (pos: (0, 2), label: [BatchNorm], fill: mg),
        (pos: (0, 3), label: [ReLU\n$max(0, dot)$], fill: mr),
        (pos: (0, 4), label: [Dropout\n$p=0.5$], fill: mp),
        (pos: (0, 5), label: [Linear\n$W_2 dot + mb(b)_2$], fill: mo),
        (pos: (0, 6), label: [Softmax], fill: mg),
        (pos: (0, 7), label: [Output\n$hat(mb(y))$], fill: mb),
      )

      for (i, layer) in layers.enumerate() {
        fl.node(
          layer.pos,
          layer.label,
          shape: rect,
          corner-radius: 5pt,
          fill: gradient.linear(white, layer.fill, angle: 45deg),
          width: 30mm,
          height: 10mm
        )

        if i < layers.len() - 1 {
          fl.edge(layer.pos, layers.at(i + 1).pos, "->", stroke: 2pt)
        }
      }

      // Annotazioni
      fl.node((1.5, 1.5), [Trasformazione\nlineare], stroke: none, fill: none)
      fl.node((1.5, 3.5), [Non-linearità], stroke: none, fill: none)
      fl.node((1.5, 4), [Regolarizzazione], stroke: none, fill: none)
    }
  )
  ```
  caption: [Architettura tipica di una rete neurale feedforward con BatchNorm, ReLU e Dropout]
)
*/
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

== Funzioni di Loss

Le *funzioni di loss* (o funzioni di costo) quantificano l'errore tra le predizioni del modello e i valori reali. Durante il training, l'obiettivo è minimizzare questa funzione.

=== Mean Squared Error (MSE) Loss

Calcola la *differenza quadratica media* tra valori predetti e valori reali:
$
  L_"MSE" = 1/N sum_(i=1)^N (y_i - hat(y)_i)^2
$

*Proprietà*:
- Penalizza fortemente gli errori grandi (peso quadratico)
- Molto sensibile agli *outlier*
- Usata principalmente in problemi di *regressione*
- Funzione *convessa*, essenziale per l'ottimizzazione

#esempio()[
  Se $y_i = 5$ e $hat(y)_i = 3$, l'errore per questo esempio è $(5-3)^2 = 4$.
  Se $y_i = 5$ e $hat(y)_i = 1$, l'errore diventa $(5-1)^2 = 16$ (4 volte maggiore pur avendo raddoppiato la distanza).
]

=== Mean Absolute Error (MAE) Loss o L1 Loss

Misura la *differenza assoluta media* tra predizioni e valori reali:
$
  L_"MAE" = 1/N sum_(i=1)^N |y_i - hat(y)_i|
$

*Proprietà*:
- Meno sensibile agli outlier rispetto a MSE
- Penalizza linearmente gli errori
- Funzione *convessa*
- Usata in problemi di *regressione* quando si vuole robustezza agli outlier

=== Binary Cross-Entropy (BCE) Loss

La *Binary Cross-Entropy* è utilizzata per problemi di *classificazione binaria* ($y_i in {0,1}$):
$
  L_"BCE" = -1/N sum_(i=1)^N [y_i log(hat(y)_i) + (1-y_i) log(1-hat(y)_i)]
$

#nota()[
  Il modello deve produrre valori $hat(y)_i in [0,1]$, tipicamente ottenuti con una funzione *Sigmoid* all'output.
]

==== Concetto di Entropia

L'*entropia* è una misura informazionale che quantifica il "tasso di sorpresa" o l'incertezza di una distribuzione di probabilità.

#nota()[
  *Interpretazione*: L'entropia misura quanto è "caotica" una distribuzione.
  - *Alta entropia*: distribuzione uniforme (massima incertezza)
  - *Bassa entropia*: distribuzione concentrata su pochi valori (bassa incertezza)
]

#esempio()[
  Consideriamo la distribuzione delle lettere in un testo italiano:
  - Se tutte le lettere hanno la stessa probabilità $P = 1/21$: *alta entropia* (massima sorpresa)
  - Se alcune lettere sono molto frequenti (es. 'e', 'a', 'i'): *bassa entropia* (minore sorpresa)

  Quando leggiamo una parola lettera per lettera, l'entropia quantifica la nostra "sorpresa" nel vedere la lettera successiva.
]

==== Funzionamento della BCE

Nella BCE per classificazione binaria:
- Quando $y_i = 1$ (label vera): vogliamo $hat(y)_i approx 1$, quindi minimizziamo $-log hat(y)_i$
- Quando $y_i = 0$ (label vera): vogliamo $hat(y)_i approx 0$, quindi minimizziamo $-log(1 - hat(y)_i)$

La loss *penalizza fortemente* predizioni sbagliate con alta confidenza.

== Classificazione Multiclasse

Nella *classificazione multiclasse* (più di 2 classi), il modello produce diversi valori (chiamati *logits*) in uscita, uno per ogni classe $k in {1, 2, dots, K}$.

#nota()[
  I *logits* sono valori reali arbitrari (non normalizzati) prodotti dall'ultimo layer lineare del modello. Per convertirli in probabilità valide, applichiamo la funzione *Softmax*.
]

=== Softmax

La funzione *Softmax* converte i logits in una distribuzione di probabilità:
$
  "softmax"(z_i) = (e^(z_i))/(sum_(j=1)^K e^(z_j))
$

dove $K$ è il numero di classi e $z_i$ è il logit per la classe $i$.

#nota()[
  La Softmax garantisce che:
  - Ogni output sia $in [0,1]$
  - La somma di tutte le probabilità sia uguale a $1$: $sum_(i=1)^K "softmax"(z_i) = 1$
  - Ogni valore rappresenta $P(y = k | x)$, la probabilità che $x$ appartenga alla classe $k$
  - I logits più alti producono probabilità maggiori (enfasi esponenziale)
]

#esempio()[
  Dato un vettore di logits per 3 classi: $z = [2.0, 1.0, 0.1]$

  Calcolo Softmax:
  - $e^(2.0) approx 7.39$
  - $e^(1.0) approx 2.72$
  - $e^(0.1) approx 1.11$
  - Somma: $7.39 + 2.72 + 1.11 = 11.22$

  Probabilità:
  - $P(y=1|x) = 7.39/11.22 approx 0.66$
  - $P(y=2|x) = 2.72/11.22 approx 0.24$
  - $P(y=3|x) = 1.11/11.22 approx 0.10$
]

=== Verosimiglianza e Log-Likelihood

Dato un dataset $D = {(x_i, y_i)}_(i=1)^N$ con label $y_i$ e input $x_i$, consideriamo un modello $hat(y)_i = M_theta(x_i)$.

==== Caso Binario (Richiamo)

Per un problema binario $y_i in {0, 1}$ con modello parametrizzato da $theta$ che produce $hat(y)_i = M_theta (x_i) in [0,1]$, la *verosimiglianza* (likelihood) del dataset $D$ dato il modello è:
$
  P(D | theta) = product_(i=1)^N hat(y)_i^(y_i) (1-hat(y)_i)^(1-y_i)
$

#nota()[
  *Interpretazione*:
  - Quando $y_i = 1$: contributo $hat(y)_i$ (vogliamo $hat(y)_i$ vicino a 1)
  - Quando $y_i = 0$: contributo $(1-hat(y)_i)$ (vogliamo $hat(y)_i$ vicino a 0)
  - La verosimiglianza è alta quando il modello predice correttamente tutti gli esempi
]

==== Perché usiamo la Log-Likelihood?

Usiamo la *log-likelihood* per diversi motivi pratici e numerici:

+ *Stabilità numerica*: La produttoria diventa somma
  $
    log P(D | theta) = log product_(i=1)^N p_i = sum_(i=1)^N log p_i
  $

+ *Evitare underflow*: Con $N$ grande, prodotto di probabilità $< 1$ tende rapidamente a 0

+ *Ottimizzazione più semplice*: Derivate della somma sono più facili da calcolare

+ *Interpretazione additiva*: Contributi indipendenti di ogni esempio

#esempio()[
  *Senza log*: $0.9 times 0.8 times 0.7 times dots times 0.6$ con 100 termini $->$ underflow

  *Con log*: $log 0.9 + log 0.8 + log 0.7 + dots + log 0.6$ $->$ stabile
]

La log-likelihood per il caso binario diventa:
$
  log P(D | theta) = sum_(i=1)^N [y_i log hat(y)_i + (1-y_i) log(1-hat(y)_i)]
$

==== Caso Multiclasse

Per la classificazione multiclasse con $y_i in {1, 2, dots, K}$, il modello produce un vettore di probabilità $hat(mb(y))_i = (hat(y)_(i,1), hat(y)_(i,2), dots, hat(y)_(i,K))$ dove $hat(y)_(i,k) = P(y_i = k | x_i)$.

La *verosimiglianza* diventa:
$
  P(D | theta) = product_(i=1)^N P(y_i | x_i) = product_(i=1)^N hat(y)_(i,y_i)
$

Dove $hat(y)_(i,y_i)$ indica la probabilità assegnata alla classe corretta per l'esempio $i$-esimo.

La *log-likelihood* diventa:
$
  log P(D | theta) = sum_(i=1)^N log P(y_i | x_i) = sum_(i=1)^N log hat(y)_(i,y_i)
$

=== Cross-Entropy Loss per Classificazione Multiclasse

La *Cross-Entropy Loss* (o *Categorical Cross-Entropy*) è definita come la *Negative Log-Likelihood* normalizzata:
$
  L_"CE" = -1/N sum_(i=1)^N log P(y_i | x_i) = -1/N sum_(i=1)^N log hat(y)_(i,y_i)
$

dove $y_i in {1, 2, dots, K}$ è l'indice della classe corretta per l'esempio $i$-esimo e $hat(y)_(i,y_i)$ è la probabilità predetta per quella classe.

#nota()[
  *Minimizzare la Cross-Entropy* equivale a *massimizzare la log-likelihood*:
  $
    min_theta L_"CE" equiv max_theta log P(D | theta)
  $

  Cerchiamo quindi i parametri $theta^*$ che meglio spiegano il dataset $D$, considerando i bias induttivi dell'architettura scelta.
]

==== One-Hot Encoding

Le label $y_i in {1, 2, dots, K}$ vengono spesso rappresentate in *one-hot encoding* per facilitare il calcolo:
- Un vettore $mb(y)_i in RR^K$ con tutti $0$ tranne un $1$ in posizione $y_i$ (classe corretta)
- Esempio: per $K=3$ e classe $y_i = 2$: $mb(y)_i = [0, 1, 0]$

*Formula alternativa con one-hot encoding*:
$
  L_"CE" = -1/N sum_(i=1)^N sum_(k=1)^K mb(y)_(i,k) log hat(y)_(i,k)
$

Dove:
- $mb(y)_(i,k) = cases(1 "se" k = y_i, 0 "altrimenti")$
- La somma interna seleziona automaticamente solo la classe corretta

#esempio()[
  Supponiamo $K = 3$ classi e l'esempio $i$-esimo ha label vera $y_i = 2$:

  *One-hot encoding*: $mb(y)_i = [0, 1, 0]$

  *Output Softmax*: $hat(mb(y))_i = [0.1, 0.7, 0.2]$ (dopo normalizzazione)

  *Calcolo loss*:
  $
    L_i = -(0 times log 0.1 + 1 times log 0.7 + 0 times log 0.2) = -log 0.7 approx 0.357
  $

  La loss considera solo: $-log hat(y)_(i,2) = -log 0.7$

  *Obiettivo*: vogliamo $hat(y)_(i,2) approx 1$ e $hat(y)_(i,1), hat(y)_(i,3) approx 0$
]

#nota()[
  Il *one-hot encoding* agisce come una *maschera* che seleziona solo la probabilità predetta per la classe corretta. Questa rappresentazione:
  - Facilita il calcolo dei gradienti durante il backpropagation
  - Rende uniforme il trattamento di tutte le classi
  - Permette un'implementazione vettoriale efficiente
]

#attenzione()[
  *Obiettivo finale del training*:

  Vogliamo trovare i parametri ottimali $theta^*$ che:
  $
    theta^* = arg max_theta log P(D | theta) equiv arg min_theta L_"CE"
  $

  Questo corrisponde a trovare il modello che:
  - *Spiega meglio* il dataset $D$ osservato
  - Sfrutta i *bias induttivi* introdotti nell'architettura (layer, attivazioni, profondità)
  - *Generalizza* bene su dati non visti (validation/test set)
]

== Training e Ottimizzazione

=== Inizializzazione dei Pesi

All'inizio del training, i *parametri del modello* (pesi e bias) devono essere inizializzati. L'inizializzazione è *fondamentale* per il successo dell'addestramento.

#attenzione()[
  Una cattiva inizializzazione può:
  - Far convergere il modello a minimi locali subottimali
  - Causare *vanishing/exploding gradients*
  - Rallentare significativamente il training
  - Impedire completamente l'apprendimento
]

==== Strategie di Inizializzazione

*Inizializzazione casuale semplice*: Non consigliata!
- Valori troppo grandi: gradienti esplodono
- Valori troppo piccoli: gradienti svaniscono
- Tutti uguali (es. zero): simmetria non rotta, neuroni imparano la stessa cosa

*Inizializzazioni avanzate*:

#esempio()[
  *Xavier/Glorot Initialization* (per tanh, sigmoid):

  I pesi di un layer con $n_"in"$ input e $n_"out"$ output sono inizializzati da:
  $
    W tilde cal(N)(0, sigma^2) "dove" sigma = sqrt(2/(n_"in" + n_"out"))
  $

  In PyTorch:
  ```python
  nn.init.xavier_normal_(layer.weight)
  # oppure uniforme:
  nn.init.xavier_uniform_(layer.weight)
  ```
]

#esempio()[
  *He Initialization* (per ReLU e varianti):

  Ottimizzata per funzioni di attivazione ReLU:
  $
    W tilde cal(N)(0, sigma^2) "dove" sigma = sqrt(2/n_"in")
  $

  In PyTorch:
  ```python
  nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
  ```
]

#nota()[
  *Perché queste inizializzazioni funzionano?*

  Mantengono la *varianza delle attivazioni* e dei *gradienti* costante attraverso i layer:
  - Evitano il vanishing gradient (segnale troppo piccolo)
  - Evitano l'exploding gradient (segnale troppo grande)
  - Permettono un flusso stabile di informazione avanti e indietro
]

==== Inizializzazione dei Bias

I bias sono tipicamente inizializzati a *zero*:
```python
nn.init.zeros_(layer.bias)
```

In alcuni casi specifici:
- LSTM/GRU: bias dei gate di forget inizializzati a 1
- Output layer: bias può essere inizializzato in base alle frequenze delle classi

=== Processo di Training

Il processo di training di una rete neurale segue questi passi:

+ *Forward pass*: calcolo delle predizioni attraverso i layer
  $
    hat(mb(y)) = f(mb(x); theta)
  $

+ *Calcolo della loss*: valutazione dell'errore
  $
    L = L(hat(mb(y)), mb(y))
  $

+ *Backward pass*: calcolo dei gradienti tramite backpropagation
  $
    nabla_theta L
  $

+ *Aggiornamento parametri*: ottimizzazione (es. SGD, Adam)
  $
    theta <- theta - eta nabla_theta L
  $

/*
#figura(
  ```python
  import fletcher as fl

  fl.diagram(
    node-stroke: 1pt,
    node-fill: gradient.linear(..mo.colors),
    spacing: (15mm, 10mm),
    edge-stroke: 1pt,
    {
      let (input, forward, loss, backward, update, output) = ((0,0), (1,0), (2,0), (2,1), (1,1), (0,1))

      fl.node(input, [Input \ $mb(x), mb(y)$], corner-radius: 5pt, extrude: (0, 3))
      fl.node(forward, [Forward \ Pass], corner-radius: 5pt)
      fl.node(loss, [Loss \ $L$], corner-radius: 5pt, fill: mr)
      fl.node(backward, [Backward \ Pass], corner-radius: 5pt)
      fl.node(update, [Update \ $theta$], corner-radius: 5pt, fill: mg)

      fl.edge(input, forward, "->", [Batch])
      fl.edge(forward, loss, "->", [$hat(mb(y))$])
      fl.edge(loss, backward, "->", [Gradiente])
      fl.edge(backward, update, "->", [$nabla_theta L$])
      fl.edge(update, forward, "->", [Parametri\naggiornati], label-pos: 0.3)
    }
  )
  ```
  caption: [Ciclo di training di una rete neurale: forward pass, calcolo loss, backward pass, aggiornamento parametri]
)
*/
#nota()[
  PyTorch automatizza il calcolo dei gradienti (punto 3) tramite *autograd*, permettendo di concentrarsi sulla definizione del modello e della loss.
]

=== Ottimizzatori

Gli *ottimizzatori* implementano algoritmi per aggiornare i parametri $theta$ in base ai gradienti calcolati.

==== Stochastic Gradient Descent (SGD)

L'ottimizzatore più semplice:
$
  theta_(t+1) = theta_t - eta nabla_theta L(theta_t)
$

dove $eta$ è il *learning rate*.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

*Variante con momentum*:
$
      v_(t+1) & = gamma v_t + eta nabla_theta L(theta_t) \
  theta_(t+1) & = theta_t - v_(t+1)
$

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

#nota()[
  Il *momentum* aiuta ad accelerare la convergenza e superare minimi locali accumulando una "velocità" nella direzione dei gradienti passati.
]

==== Adam (Adaptive Moment Estimation)

Uno degli ottimizzatori più usati, combina:
- *Momentum*: media mobile dei gradienti
- *RMSprop*: adatta il learning rate per ogni parametro

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

*Vantaggi di Adam*:
- Convergenza veloce
- Poco sensibile alla scelta del learning rate iniziale
- Adatta automaticamente il learning rate per ogni parametro
- Funziona bene nella maggior parte dei casi

#esempio()[
  *Setup tipico per training*:
  ```python
  model = MyModel()
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  for epoch in range(num_epochs):
      for batch_x, batch_y in dataloader:
          # Forward pass
          outputs = model(batch_x)
          loss = criterion(outputs, batch_y)

          # Backward pass
          optimizer.zero_grad()  # Reset gradienti
          loss.backward()        # Calcolo gradienti
          optimizer.step()       # Aggiornamento parametri
  ```
]

==== Altri Ottimizzatori

*AdamW*: Variante di Adam con weight decay corretto
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

*RMSprop*: Adatta il learning rate usando media mobile dei gradienti al quadrato
```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
```

#attenzione()[
  *Scelta dell'ottimizzatore*:
  - *Adam/AdamW*: scelta sicura per la maggior parte dei problemi
  - *SGD con momentum*: può generalizzare meglio con tuning accurato
  - *RMSprop*: buono per RNN e problemi con gradienti variabili
]

=== Learning Rate e Scheduling

Il *learning rate* $eta$ è uno degli iperparametri più importanti:
- Troppo alto: divergenza, oscillazioni
- Troppo basso: convergenza lenta, minimi locali

==== Learning Rate Scheduling

Modificare il learning rate durante il training:

*Step Decay*: Riduce $eta$ ogni $N$ epoche
```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

*Cosine Annealing*: Riduce $eta$ seguendo una curva coseno
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

*ReduceLROnPlateau*: Riduce $eta$ quando la loss smette di migliorare
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=0.5, patience=5)
```

#esempio()[
  *Uso dello scheduler*:
  ```python
  for epoch in range(num_epochs):
      train_one_epoch(model, optimizer, criterion, train_loader)
      val_loss = validate(model, criterion, val_loader)

      # Aggiorna learning rate
      scheduler.step(val_loss)  # per ReduceLROnPlateau
      # oppure scheduler.step() per altri scheduler
  ```
]

== Regolarizzazione e Prevenzione dell'Overfitting

Le tecniche di *regolarizzazione* aiutano il modello a *generalizzare* meglio su dati non visti, prevenendo l'*overfitting*.

=== Weight Decay (L2 Regularization)

Aggiunge un termine di penalizzazione alla loss per limitare la magnitudine dei pesi:
$
  L_"total" = L_"task" + lambda/2 sum_i theta_i^2
$

dove $lambda$ è il *coefficiente di regolarizzazione*.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
```

#nota()[
  *Effetto*: pesi più piccoli rendono il modello meno sensibile a piccole variazioni nell'input, migliorando la generalizzazione.
]

=== Dropout

Durante il training, *disattiva casualmente* una frazione $p$ di neuroni:
- Ogni neurone ha probabilità $p$ di essere "spento" (output = 0)
- A ogni iterazione, diversi neuroni sono attivi
- Durante l'inferenza, tutti i neuroni sono attivi (con scaling)

/*
```python
class ModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(p=0.5)  # 50% dropout
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.3)  # 30% dropout
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Applicato solo in training
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)
```
*/

#nota()[
  *Perché funziona?*
  - Impedisce ai neuroni di co-adattarsi troppo
  - Equivale a fare ensemble di tante reti diverse
  - Riduce l'overfitting forzando ridondanza
]

#attenzione()[
  Ricordarsi di mettere il modello in modalità corretta:
  ```python
  model.train()  # Training: dropout attivo
  model.eval()   # Inferenza: dropout disattivo
  ```
]

=== Batch Normalization

Normalizza le attivazioni di ogni layer durante il training:
$
  hat(x) = (x - mu_cal(B))/sqrt(sigma_cal(B)^2 + epsilon)
$

dove $mu_cal(B)$ e $sigma_cal(B)$ sono media e varianza del batch.

```python
class ModelWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        return self.fc3(x)
```

*Vantaggi*:
- Permette learning rate più alti
- Riduce la dipendenza dall'inizializzazione
- Ha effetto regolarizzante
- Accelera la convergenza

#nota()[
  *Ordine tipico dei layer*:
  ```
  Linear -> BatchNorm -> Activation (ReLU) -> Dropout
  ```

  (Anche se l'ordine BatchNorm/Activation è dibattuto)
]

=== Early Stopping

Interrompe il training quando la *validation loss* smette di migliorare:

```python
best_val_loss = float('inf')
patience = 10  # Numero di epoche senza miglioramento
patience_counter = 0

for epoch in range(max_epochs):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Salva il modello migliore
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

#nota()[
  *Previene overfitting* interrompendo prima che il modello cominci a memorizzare i dati di training.
]

=== Data Augmentation

Per problemi di visione, aumenta artificialmente il dataset:
- Rotazioni, traslazioni, flip
- Crop casuali
- Variazioni di colore/contrasto
- Mixup, CutMix (tecniche più avanzate)

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

== Monitoraggio e Debugging

=== Metriche di Valutazione

Oltre alla loss, monitoriamo metriche interpretabili:

*Classificazione*:
- *Accuracy*: percentuale di predizioni corrette
- *Precision, Recall, F1-score*: per classi sbilanciate
- *Confusion Matrix*: errori per classe

*Regressione*:
- *MAE, RMSE*: errori medi
- *R²*: varianza spiegata

```python
def compute_accuracy(outputs, labels):
    _, predictions = torch.max(outputs, dim=1)
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0)
```

=== Curve di Apprendimento

Visualizzare training e validation loss/accuracy:

#nota()[
  *Interpretazione delle curve*:
  - *Underfitting*: train e val loss entrambe alte
  - *Overfitting*: train loss bassa, val loss alta e crescente
  - *Buon fit*: train e val loss entrambe basse e vicine
]

=== Problemi Comuni

*Loss non diminuisce*:
- Learning rate troppo basso o alto
- Inizializzazione sbagliata
- Bug nel codice (gradienti non calcolati)
- Normalizzazione dati mancante

*Loss diventa NaN*:
- Learning rate troppo alto (gradienti esplodono)
- Divisione per zero o log(0)
- Overflow numerico

*Overfitting*:
- Modello troppo complesso per i dati
- Dati di training insufficienti
- Mancano tecniche di regolarizzazione

#attenzione()[
  *Debugging checklist*:
  + Verifica shape dei tensori
  + Controlla che i gradienti siano calcolati (`loss.backward()`)
  + Verifica che l'ottimizzatore aggiorni i parametri
  + Testa su un batch piccolo (overfit intenzionale)
  + Visualizza le attivazioni e i gradienti
]
