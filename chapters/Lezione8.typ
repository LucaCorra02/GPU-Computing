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

== Classificatori lineari

Per capire il significato di avere dei modelli _profondi_ (più layer nascosti), andiamo a introdurre i *classificatori lineari*.

Un *classificatore lineare binario* è un modello che prende un input $x in R^D$ (vettore di dimensione $D$), lo moltiplica per dei pesi $w in R^D$ e aggiunge un bias $b in R$. Il risultato viene passato a una funzione di attivazione $mr(sigma)$:
$
  x -> mr(sigma)(w dot x + b)
$
#nota()[
  La formula rappresenta il lavoro di un singolo neurone
]
Se volessimo estendere il classificatore a *un classificatore multiclass* (più classi possibili di *output*), dovremo passare da un vettore dei pesi $w$ a una matrice dei pesi $W$:
$
  x -> sigma(W x + b)
$
Dove la funzione di attivazione viene è applicata *componente per componente*.

#figure(
  image("/assets/linear-classifier.png", width: 70%),
)

Nell'immagine di sinistra, tutti gli input $x$ convergono in un singolo neurone con funzione di attivazione $sigma$, producendo un singolo output. A destra, gli input si connettono a multipli neuroni output (uno per classe), ciascuno con la propria funzione di attivazione $sigma$, permettendo la classificazione in $K$ classi diverse..

#attenzione()[
  Questo tipo di architettura semplice *non* è funzionale, permette solamente di imparare relazioni lineari (o combinazioni lineari) rispetto all'input grezzo. Non è in grado di imparare caratteristiche complesse o gerarchiche.
]

#informalmente()[
  Se i dati del dataset, sono divisi in due categorie, dove: i blu sono al centro e i dati rossi sono intorno (come un cerchio), il classificatore lineare fallirà sempre, indipendentemente dalla funzione di attivazione in uscita. L'equazione della retta è data da $x dot w + b = 0$ troppo semplice in questo caso.
]

== Multi-Layer Perceptron (MLP)

Per questo motivo si passa a dei modelli con un architettura gerarchica organizzata su livelli. I modelli MLP presentano un *architettura a strati*, dove:
- Ogni MLP è composto da L strati, dove ogni strato presenta i propri pesi $W^l$ e $b^l$.
- L'output di uno strato diventa l'input del successivo:
$
  x^l = sigma(W^l dot x^(l-1) + b^l)
$
Dove $x^0$ è l'input della rete originale. L'output della rete è dunque:
$
  f(x,W,b) = x^L
$

#figure(
  image("/assets/mlp.png", width: 70%),
)

Le reti neurali profonde trasformano i dati attraverso *successive mappature non lineari*:
$
  x -> x^1 -> x^2 -> dots -> x^L
$

Ogni livello produce una *nuova rappresentazione* dell'input:
- *Obiettivo*: rendere il compito finale (classificazione, regressione) più facile
- Le rappresentazioni intermedie catturano feature sempre più astratte
- L'ultimo livello opera in uno spazio dove le classi sono (*idealmente*) linearmente separabili

#informalmente()[
  Un singolo strato è solo una trasformazione lineare seguita da una non-linearità. Usare più strati permette la composizione di funzioni *non lineari*, abilitando il modello a rappresentare funzioni molto più complesse.
]

=== Embedding Space

L'output di un layer nascosto definisce uno *spazio di embedding*: Un buon embedding deve avere le segunti Proprietà:
- *Cattura la struttura* dei dati (punti simili sono vicini nello spazio)
- *Separa le classi* (punti di classi diverse sono distanti)
- *Semplifica le decisioni* downstream (il classificatore finale diventa più semplice)


Ogni input $x$ viene mappato in questo spazio tramite forward propagation:
$
  x -> "embedding"(x) = x^l = sigma(W^l dot sigma(W^(l-1) dot (dots)))
$

#informalmente()[
  Uno spazio di embedding ben progettato trasforma il problema originale (magari non linearmente separabile) in uno spazio dove diventa linearmente separabile o molto più semplice da risolvere.
]

== Funzioni di Attivazione

I modelli di deep learning sono in grado di catturare informazioni _al di là dell'osservabile_, grazie all'uso delle *funzioni di attivazione* che introducono non-linearità nel modello.

=== Caratteristiche delle Funzioni di Attivazione

Le *funzioni di attivazione* presentano caratteristiche specifiche:
- Hanno tipicamente *due asintoti* (da $+infinity$ a $-infinity$)
- Per ogni valore $x$ tendono ad essere in *saturazione* tra $-1$ e $1$ (o $0$ e $1$ a seconda della funzione)
- La *parte centrale* rappresenta la regione di incertezza del modello

#nota()[
  Le funzioni di attivazione servono per definire *regioni non lineari* nello spazio, creando curvature che permettono di separare gruppi di dati in base alla loro categoria di appartenenza (in uno *stesso spazio di embedding*) creando superfici di separazione complesse.
]

=== Tangente Iperbolica (Tanh)

La *Tangente iperbolica* ($tanh$) è una delle funzioni di attivazione classiche, definita come:
$
  tanh(x) = (2)/(1+e^(-2x))-1
$
Ha output nel range $[-1, 1]$ con saturazione agli estremi (per $x$ di dimensioni elevate) e maggiore sensibilità al centro.

#figure(
  cetz.canvas({
    import cetz.draw: *

    // Assi
    line((-3.5, 0), (3.5, 0), mark: (end: ">"))
    content((3.7, 0), $x$)
    line((0, -1.5), (0, 1.5), mark: (end: ">"))
    content((0, 1.7), $tanh(x)$)

    // Tacche asse x
    for i in range(-3, 4) {
      if i != 0 {
        line((i, -0.05), (i, 0.05))
        content((i, -0.25), text(size: 8pt, str(i)))
      }
    }

    // Tacche asse y
    for i in (-1, -0.5, 0.5, 1) {
      line((-0.05, i), (0.05, i))
      content((-0.3, i), text(size: 8pt, if i == 0.5 { "0.5" } else if i == -0.5 { "-0.5" } else { str(i) }))
    }

    // Linee tratteggiate per limiti
    line((-3.5, 1), (3.5, 1), stroke: (paint: gray, dash: "dashed"))
    line((-3.5, -1), (3.5, -1), stroke: (paint: gray, dash: "dashed"))

    // Curva tanh
    let points = ()
    for i in range(0, 101) {
      let x = -3 + i * 6 / 100
      let y = calc.tanh(x)
      points.push((x, y))
    }
    line(..points, stroke: (paint: blue, thickness: 2pt))
  }),
  caption: [Grafico della funzione tangente iperbolica $tanh(x)$],
)


=== ReLU (Rectified Linear Unit)

La *ReLU* è una delle funzioni di attivazione più utilizzate. È definita come:
$
  "ReLU"(x) = max(0, x)
$

#nota()[
  La ReLU restituisce $0$ quando $x < 0$ e $x$ quando $x >= 0$. Applicando una trasformazione lineare si ottiene: $R(w x + b)$ che dipende dal valore della retta $w x + b$.
]

#figure(
  cetz.canvas({
    import cetz.draw: *

    // Assi
    line((-3.5, 0), (3.5, 0), mark: (end: ">"))
    content((3.7, 0), $x$)
    line((0, -0.5), (0, 3.5), mark: (end: ">"))
    content((0, 3.8), $"ReLU"(x)$)

    // Tacche asse x
    for i in range(-3, 4) {
      if i != 0 {
        line((i, -0.05), (i, 0.05))
        content((i, -0.3), text(size: 8pt, str(i)))
      }
    }

    // Tacche asse y
    for i in range(1, 4) {
      line((-0.05, i), (0.05, i))
      content((-0.3, i), text(size: 8pt, str(i)))
    }

    // Parte x < 0 (costante a 0)
    line((-3.5, 0), (0, 0), stroke: (paint: blue, thickness: 2pt))

    // Parte x >= 0 (identità)
    line((0, 0), (3, 3), stroke: (paint: blue, thickness: 2pt))

    // Punto in (0,0) per evidenziare il cambio
    circle((0, 0), radius: 0.08, fill: blue, stroke: none)
  }),
  caption: [Grafico della funzione ReLU (Rectified Linear Unit)],
)

==== Teorema di Approssimazione Universale con ReLU

Data una qualsiasi funzione continua $f in C([a,b], RR)$, è possibile *approssimala come una combinazione lineare di ReLU*:
$
  f(x) tilde sum_i sigma(w_i dot x + b_i)
$

Il principio di funzionamento è il seguente:
+ Prendiamo tante ReLU con diverse traslazioni e scalature
+ Combinandole opportunamente, otteniamo una *curva spezzata* che approssima $f$
+ Più ReLU utilizziamo, migliore è l'approssimazione (ma aumenta la complessità)

#esempio()[
  #figure(
    cetz.canvas({
      import cetz.draw: *

      // Assi
      line((-0.5, 0), (9, 0), mark: (end: ">"))
      content((9.2, 0), $x$)
      line((0, -2), (0, 2.5), mark: (end: ">"))
      content((0, 2.7), $f(x)$)

      // Griglia leggera
      for i in range(1, 9) {
        line((i, -2), (i, 2.5), stroke: (paint: gray.lighten(60%), thickness: 0.5pt))
      }
      for i in (-2, -1, 1, 2) {
        line((0, i), (9, i), stroke: (paint: gray.lighten(60%), thickness: 0.5pt))
      }

      // Funzione target complessa (blu scuro)
      let target = (
        (0, -0.5),
        (0.5, -1.2),
        (1, -0.8),
        (1.5, 0.2),
        (2, 1.3),
        (2.5, 1.5),
        (3, 1.2),
        (3.5, 0.6),
        (4, 0.3),
        (4.5, 0.5),
        (5, 1.5),
        (5.5, 2.0),
        (6, 1.8),
        (6.5, 0.8),
        (7, 0.2),
        (7.5, -0.5),
        (8, -1.2),
        (8.5, -1.5),
      )
      line(..target, stroke: (paint: blue.darken(20%), thickness: 3pt))

      // ReLU individuali (grigio chiaro) - alcune componenti
      // ReLU 1
      line((0, -0.8), (1.5, -0.8), stroke: (paint: gray, thickness: 1.5pt))
      line((1.5, -0.8), (3, 0.7), stroke: (paint: gray, thickness: 1.5pt))

      // ReLU 2
      line((3, 0.3), (4.5, 0.3), stroke: (paint: gray, thickness: 1.5pt))
      line((4.5, 0.3), (6, 1.8), stroke: (paint: gray, thickness: 1.5pt))

      // ReLU 3
      line((6, 1.5), (7, 1.5), stroke: (paint: gray, thickness: 1.5pt))
      line((7, 1.5), (8.5, 0), stroke: (paint: gray, thickness: 1.5pt))

      // Una ReLU evidenziata in rosso
      line((6.5, 0), (7.5, 0), stroke: (paint: red, thickness: 2pt))
      line((7.5, 0), (9, 1.5), stroke: (paint: red, thickness: 2pt))

      // Etichetta formula
      content((4.5, 2.3), text(
        size: 10pt,
        $f(x) = sigma(w_1 x + b_1) + sigma(w_2 x + b_2) + sigma(w_3 x + b_3) + dots$,
      ))
    }),
    caption: [Approssimazione di una $mb("funzione")$ mediante combinazione lineare di funzioni ReLU],
  )
  Le componenti *grigie* rappresentano singole ReLU traslate e scalate, mentre la ReLU $mr("rossa")$ mostra un esempio specifico. Sommando opportunamente queste funzioni si ottiene una curva spezzata che approssima la funzione target.
]

#attenzione()[
  *Limitazioni*:
  - Il modello può diventare *molto complesso* con molte unità
  - Richiede *più tempo di training* e *maggiore quantità di dati* per raggiungere un'accuratezza accettabile
  - Trade-off tra capacità di approssimazione e costo computazionale
]

=== Generalizzazione del Teorema

Il teorema può essere generalizzato: per ogni funzione continua $g:[0,1]^D -> RR$ (Con $D$ dimensione dell'input) è possibile approssimarla con *un singolo livello nascosto* (percettrone) utilizzando:
- $W in RR^(K times D)$ (matrice dei pesi)
- $b in RR^K$ (vettore di bias)
- $omega in RR^K$ (pesi di output)
La rete sarà quindi:
$
  x -> w^T sigma(W x + b)
$

#nota()[
  Aumentando $K$ (ovvero il numero di *unità nascoste*), la rete a più possibilità di apprendere funzioni complesse, ma con un costo computazionale maggiore.
]

Il vantaggio principale di avere dei *modelli profondi* (deep) rispetto ai modelli superficiali (flat) è che le *reti neurali gerarchiche* con diversi livelli di profondità sono in grado di estrarre *più informazioni strutturate* dai dati. Questo accade in quanto i modelli profondi sfruttano *bias induttivi* (scelti dal progettista) per apprendere pattern sempre più complessi attraverso i livelli.

#nota()[
  Un *bias induttivo* inferisce sull'architettura della rete, assumento che l'output deriva da una specifica composizione dell'input
]

#esempio()[
  *Visione artificiale*:
  - *Primi livelli*: catturano pattern basilari come edge, angoli, texture
  - *Livelli intermedi*: combinano i pattern basilari in strutture più complesse (parti di oggetti)
  - *Ultimi livelli*: comprendono l'immagine nel suo complesso, riconoscendo oggetti e scene
]

Il *learning* è una stima di parametri sempre più efficaci, utilizzando bias induttivi (embedding e livelli intermedi) per favorire l'apprendimento di *rappresentazioni gerarchiche* dei dati.

#attenzione()[
  Sebbene il teorema di approssimazione universale sia perfetto da un pinto di vista teorico, presenta alcune *limitazioni importanti*:
  - *Garantisce* un basso errore di training (la rete può approssimare qualsiasi funzione continua)
  - *Non dice nulla* sulla capacità di generalizzazione (performance su dati mai visti)
  - Aumentare $K$ (numero di unità nascoste) migliora sempre il fit sul training set
  - *Rischio*: *overfitting* se il modello è troppo complesso rispetto ai dati disponibili
]

=== Bilanciamento nel Training Reale

Nel training reale, l'apprendimento richiede un *bilanciamento* tra due estremi:

/ Underfitting (modello troppo semplice): Il modello non ha abbastanza capacità per catturare la struttura dei dati. Ha alto errore sia sul training set che sul test set.

/ Overfitting (modello troppo complesso): Il modello memorizza i dati di training invece di apprendere pattern generalizzabili. Ha basso errore sul training set ma alto errore sul test set.

#nota()[
  La *profondità* (depth) della rete gioca un ruolo cruciale in questo bilanciamento. Reti più profonde hanno maggiore capacità espressiva, ma richiedono più dati e tecniche di regolarizzazione per evitare overfitting.
]

== PyTorch nn Framework

Verrà utilizzato il modulo `torch.nn`, per costruire modelli di deep learning in PyTorch.

Il framework `torch.nn` fornisce diversi moduli:
/ Layer: Tutti i vari *livelli* della rete. Esistono layer ricorrenti, lineari, convoluzionali, ecc.
/ Funzioni di attivazione: ReLU, Sigmoid, Tanh, Softmax, ecc.
/ Livelli di normalizzazione: BatchNorm, LayerNorm, ecc.
/ Funzioni di perdita (loss): MSE, CrossEntropy, BCE, ecc.

=== Modules (Layers)

Quando creiamo un nuovo modello, dobbiamo costruire una *sotto-classe di `nn.Module`*.


 La struttura base è:

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
