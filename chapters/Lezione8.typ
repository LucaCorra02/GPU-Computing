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
La stuttura è la seguente:
- `__init__`: deifinisce l'architettura della rete neurale (numero di later del modello)
- `forward()`: implementa come computare i logits a partire dall'output

#attenzione()[
  In PyTorch, un modello viene chiamato direttamente come `output = model(x)`. Questo in quanto il modulo `nn.Module` sovrascrive il metodo di pythom `__call__`.

  Quando viene eseguito `y=model(x)`, vengono chiamati una serie di metodi aggiuntivi prima del metodo `forward()` (è sempre consigliato usare questa sinattsi piuttosto che chiamare direttamente il metodo `forward`).
]

=== Layer Lineari (Trasformazioni Affini)

Una classica trasformazione lineare (eseguita da un singolo neuorone):
$
  "output" = tanh(a x + b)
$
In PyTorch viene implementata seguendo una forma equivalente ottimizata che prende il nome di *trasformazione affine*. In particolare per un singolo vettore (riga) $x$:
$
  y = x A^T
$
Dove:
- _input_ $x in R^(1 times D)$
- _weight_ $A in R^(C times D)$
- _output_ $y = x A^T in R^(1 times C)$

Questa scrittura risulta equivalente alla forma classica $y = x A$, in quanto l'input $x$ è rappresentato come un vettore riga, mentre i batch come una matrice.

#nota()[
  Viene applicata la seguente trasformazione dove $overline(x)$ indica un vettore riga (anzichè a colonna):
  $
    y & = (A overline(x))^t + overline(b)^t \
    y & = overline(x)^t + A^t + overline(b)^t \
    y & = x A^t + b
  $
]


La scrittura sopra risulta adattarse meglio per i *batch*.
Dato un Batch (di dimensione $B$) chiamato $X$, ovvero un insieme di vettori $x_i$ ciascuno con dimensione $D$ e considerano $C$ dimensioni di output, otteniamo:
$
  X = vec(x_1, x_2, dots, x_n) in R^(B times D)
$
Possiamo applicare una trasformazione all'intero batch nel seguente modo:
$
  Y = X A^T, space Y in R^(B times C)
$
#nota()[
  La trasposizione è necessaria in quanto $A in R^(C times D)$, per le proprietà del prodotto matriciale le dimensioni devono combiaciare con quelle dell'input.
]

Ogni riga soddisfa:
$
  y_i = x_i A^T "for" i = 1,dots,B
$

Aggiungendo il bias $b in R^C$, la formula diventa:
$
  Y = X A^T + 1b^t
$
Dove $1 in R^B$ è un vettore di uni, in questo modo $b$ viene _trasmesso_ a tutte le righe.

#esempio()[
  Esempio di rete con 2 layer:
  - `Linear(in_features=3, out_features=5, bias=True)`
  - `Linear(in_features=5, out_features=2, bias=True)`

  ```py
  import torch
  class SimpleNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.fc1 = nn.Linear(3, 5)
      self.relu = nn.ReLU()
      self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
      x = self.fc1(x)
      x = self.relu(x)
      x = self.fc2(x)
      return x

    model = SimpleNet()
    x = torch.randn(6, 3)
    y = model(x)
  ```
  In particolare l'otput $y$ avrà size `[6,2]` (batch da 6 righe con 2 features).
]

=== Altri layer

Oltre ai clasici layer lineari, esistono diversi tipi di layer:

#figure(
  table(
    columns: 3,
    align: (left, left, left),
    stroke: 0.5pt,
    [*Layer*], [*Description*], [*Example*],
    [#raw("nn.Linear")], [Fully connected layer], [#raw("nn.Linear(in_features, out_features)")],
    [#raw("nn.Conv2d")], [2D convolution], [#raw("nn.Conv2d(3, 64, 3, stride=1, padding=1)")],
    [#raw("nn.LSTM")], [Recurrent layer], [#raw("nn.LSTM(input_size, hidden_size)")],
    [#raw("nn.BatchNorm2d")], [Normalization layer], [#raw("nn.BatchNorm2d(64)")],
  ),
  kind: table,
)

== Funzioni di Loss

Le *funzioni di loss* (o funzioni di costo) quantificano l'errore tra le predizioni del modello e i valori reali. Durante il *training*, l'obiettivo è *minimizzare questa funzione*.


#attenzione()[
  Nell'intera sezione chiameremo le predizioni del modello con $hat(y_i)$ mentre l'etichetta corretta associata con $y_i$.
]
=== Mean Squared Error (MSE) Loss

Calcola la *differenza quadratica media* tra valori predetti e valori reali:
$
  L_"MSE" = 1/N sum_(i=1)^N (y_i - hat(y)_i)^2
$
Dove $N$ è la dimensione del _corpus_. *Proprietà*:
- *Penalizza* fortemente gli *errori grandi* (peso quadratico)
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
- *Meno sensibile* agli outlier rispetto a MSE
- Penalizza linearmente gli errori
- Funzione *convessa*
- Usata in problemi di *regressione* quando si vuole robustezza agli outlier

#figure(
  cetz.canvas(length: 0.8cm, {
    import cetz.draw: *

    // Configurazione - grafico più largo e più basso
    let x-min = -6
    let x-max = 6
    let y-max = 10

    // Assi
    line((x-min, 0), (x-max, 0), mark: (end: ">"))
    content((x-max + 0.6, 0.5), text(size: 9pt, $"Errore" (y - hat(y))$))
    line((0, 0), (0, y-max), mark: (end: ">"))
    content((0, y-max + 0.6), text(size: 9pt, $"Loss"$))

    // Griglia
    for i in range(-6, 7) {
      if i != 0 and calc.rem(i, 2) == 0 {
        line((i, 0), (i, y-max), stroke: (paint: gray.lighten(70%), thickness: 0.5pt, dash: "dotted"))
        line((i, -0.15), (i, 0.15), stroke: 0.5pt)
        content((i, -0.5), text(size: 7pt, str(i)))
      }
    }

    for i in (2, 4, 6, 8, 10) {
      line((x-min, i), (x-max, i), stroke: (paint: gray.lighten(70%), thickness: 0.5pt, dash: "dotted"))
      line((-0.1, i), (0.1, i), stroke: 0.5pt)
      content((-0.5, i), text(size: 7pt, str(i)))
    }

    // Funzione MSE (parabola): y = x²/3 (scalata per entrare nel grafico)
    let mse-points = ()
    for i in range(-60, 61) {
      let x = i / 10.0
      let y = (x * x) / 3.6
      if y <= y-max {
        mse-points.push((x, y))
      }
    }
    line(..mse-points, stroke: (paint: blue, thickness: 2pt))

    // Funzione MAE (V-shape): y = |x|
    let mae-points = ()
    for i in range(-60, 61) {
      let x = i / 10.0
      let y = calc.abs(x) * 1.5
      if y <= y-max {
        mae-points.push((x, y))
      }
    }
    line(..mae-points, stroke: (paint: orange, thickness: 2pt, dash: "dashed"))

    // Punto minimo evidenziato
    circle((0, 0), radius: 0.08, fill: black, stroke: none)

    // Legenda
    rect((3.2, 8.5), (5.5, 9.8), fill: white, stroke: 0.5pt)
    line((3.4, 9.4), (4, 9.4), stroke: (paint: blue, thickness: 2pt))
    content((4.2, 9.4), text(size: 7pt, $"MSE"$), anchor: "west")
    line((3.4, 8.9), (4, 8.9), stroke: (paint: orange, thickness: 2pt, dash: "dashed"))
    content((4.2, 8.9), text(size: 7pt, $"MAE"$), anchor: "west")
  }),
  caption: [Confronto tra Mean Squared Error (MSE) e Mean Absolute Error (MAE). MSE penalizza quadraticamente gli errori (cresce rapidamente), mentre MAE penalizza linearmente (crescita costante).],
)

=== Binary Cross-Entropy (BCE) Loss

#informalmente()[
  Quanto ti sei sbagliato nel valutare la probabilità di successo e quella di fallimento.
]

La *Binary Cross-Entropy* è utilizzata per problemi di *classificazione binaria* ($y_i in {0,1}$):
$
  L_"BCE" = -1/N sum_(i=1)^N [y_i log(hat(y)_i) + (1-y_i) log(1-hat(y)_i)]
$

#nota()[
  Siccome il modello produce dei valori $hat(y)_i in (-infinity,+infinity)$ (logits), viene applicata una funzione *Sigmoid* $sigma$ all'output del modello (non esiste il $log$ di numeri negativi). La sigmoide è una funzione con una forma ad `S`, che mappa un qualsiasi valore reale in un intervallo compreso tra $[0,1]$, aderendo così alle richieste della funzione di loss.

  La funzione `nn.BCEWithLogitsLoss()` combina questi due comportamenti.
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

Nella *BCE* per classificazione binaria:
- *$y_i = 1$* (label $mg("vera")$):
  Vogliamo che il modello fornisca una predizione $hat(y)_i approx 1$:
  $
    & = -(y_i log(hat(y)_i) + (1-y_i) log(1-hat(y)_i)) \
    & = - 1 log(hat(y)_i) - (1-mg(1)) log(1-hat(y)_i) \
    & = - log(hat(y)_i)
  $
  minimizziamo $-log hat(y)_i$

- *$y_i = 0$* (label $mr("falsa")$): Vogliamo che il modello predica  $hat(y)_i approx 0$:
$
  & = -(y_i log(hat(y)_i) + (1-y_i) log(1-hat(y)_i)) \
  & = - (mr(0) log(hat(y)_i) + (1-mr(0)) log(1-hat(y)_i)) \
  & = - log(1-hat(y)_i)
$
quindi minimizziamo $-log(1 - hat(y)_i)$

La loss *penalizza fortemente* predizioni sbagliate con alta confidenza.

== Classificazione Multiclasse

Nella *classificazione multiclasse* (più di 2 classi), il modello produce diversi valori (chiamati *logits*) in uscita, uno per ogni classe $k in {1, 2, dots, K}$. Vengono quindi eseguite le seguenti operazioni:
- Si ottengono le predizioni del modello (logits)
- Viene applicata la softmax, ottenendo una probabilità per ogni possibile classe
- Viene calcolata la log-likelihood (Cross-Entropy)

#nota()[
  I *logits* sono valori reali arbitrari (non normalizzati) prodotti dall'ultimo layer lineare del modello. Per convertirli in probabilità valide, applichiamo la funzione *Softmax*.
]

Assunzioni:
- Ogni input $x$ appertiene al massimo ad una classe
- Ci sono $K$ possibili classi
- Il modello deve produrre una distribuzione di probabilità sulle $K$ classi

Formalmente il modello impara:
$
  p(y=k |x) space forall k = 1, dots, K
$
Ovvero la probabilità che $y$ appartenga alla classe $k$ dato l'input $x$.
#nota()[
  L'output del modello non sono delle probabilità, ma dei *logits*, ovveri dei valori compresi nell'intervallo $(-infinity, infinity)$.
]


=== Softmax

La funzione *Softmax* converte i logits in una distribuzione di probabilità:
$
  "softmax"(z_i) = (e^(z_i))/(sum_(j=1)^K e^(z_j))
$

dove $K$ è il numero di classi e *$z_i$* è il *logit* per la classe $i$.

#nota()[
  La Softmax garantisce che:
  - Ogni output sia $in [0,1]$
  - La somma di tutte le probabilità sia uguale a $1$: $sum_(i=1)^K "softmax"(z_i) = 1$
  - Ogni valore rappresenta $P(y = k | x)$, la probabilità che $x$ appartenga alla classe $k$
  - I logits più alti producono probabilità maggiori (*enfasi esponenziale*)
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

==== Caso Binario

Per un problema binario $y_i in {0, 1}$ con modello parametrizzato da $theta$ che produce $hat(y)_i = M_theta (x_i) in [0,1]$, la *verosimiglianza* (likelihood) del dataset $D$ dato il modello è:
$
  P(D | theta) = product_(i=1)^N hat(y)_i^(y_i) (1-hat(y)_i)^(1-y_i)
$

#nota()[
  *Interpretazione*:
  - Quando $y_i = 1$: contributo $hat(y)_i$ (vogliamo $hat(y)_i$ vicino a 1)
  - Quando $y_i = 0$: contributo $(1-hat(y)_i)$ (vogliamo $hat(y)_i$ vicino a 0)
  - *La verosimiglianza è alta quando il modello predice correttamente tutti gli esempi*
]


==== Log-Likelihoodù

#informalmente()[
  Quanto sei stato sorpreso di scoprire che la risposta giusta era quella vera
]

Viene utilizzata la *log-likelihood* per diversi motivi pratici e numerici:

+ *Stabilità numerica*: La produttoria diventa somma
  $
    log P(D | theta) underbrace(=, "Proprietà" \ "dei log") log product_(i=1)^N p_i = sum_(i=1)^N log p_i
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

Per la classificazione multiclasse con $y_i in {1, 2, dots, K}$, il modello produce un vettore di probabilità:
$
  hat(mb(y))_i = (hat(y)_(i,1), hat(y)_(i,2), dots, hat(y)_(i,K)) space "dove" hat(y)_(i,k) = P(y_i = k | x_i)
$

La *verosimiglianza* diventa:
$
  P(D | theta) = product_(i=1)^N P(y_i | x_i) = product_(i=1)^N hat(y)_(i,y_i)
$

Dove *$hat(y)_(i,y_i)$* indica la *probabilità* assegnata alla classe corretta per l'esempio $i$-esimo.

La *log-likelihood* diventa:
$
  log P(D | theta) = sum_(i=1)^N log P(y_i | x_i) = sum_(i=1)^N log hat(y)_(i,y_i)
$

=== Cross-Entropy Loss per Classificazione Multiclasse

#informalmente()[
  Quanto eri sicuro della risposta corretta rispetto a tutte le altre opzioni
]

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

  Possiamo inoltre vedere La Cross-Entropy come una composizione di due funzioni:
  $
    "CrossEntropyLoss" = "LogSoftMax" + "NegativeLogLikehood"
  $
  Tradotto in formula:
  $
    L_"CE" = -1/N sum_(i=1)^N log((e^(z_(i,y_i)))/(sum_(c=1)^C e^(z_(i,c))))
  $
]

#esempio()[
  Differenze tra BCE (binary cross entropy) e (multiclass entropy)
  ```py
    pred_prob = torch.tensor([0.9, 0.1, 0.8])
    labels = torch.tensor([1.0, 0.0, 1.0])

    bce = nn.BCELoss()
    print("BCE:", bce(logits, labels))
    #BCE: tensor(0.144), poca sorpresa

    ce = nn.CrossEntropyLoss()
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
    labels = torch.tensor([2, 1])
    print("CrossEntropy:", ce(logits, labels))
    # CrossEntropy: tensor(1.2685)
  ```
  Come si può vedere dal codice:
  - `BCE loss`: Si aspetta in input delle probabilità già calcolate (valori tra $0$ e $1$), *non* vuole i logits grezzi.\
    L'errore è basso perché le predizioni sono molto vicine alla realtà:

  - `CrossEntropy loss`: *Accetta logits grezzi* (numeri reali qualsiasi), non probabilità. Applica la Softmax internamente. Nell'esempio il modello è "molto sorpreso" (ha sbagliato clamorosamente la previsione), generando una penalità alta che alza la media dell'errore.
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
- $
  mb(y)_(i,k) = cases(1 "se" k = y_i, 0 "altrimenti")
  $
- La *somma interna seleziona automaticamente solo la classe corretta*

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

== Inizializzazione dei pesi

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

All'inizio del training, la *matrice dei pesi* (che dovrà essere appresa) viene *inizializzata casualmente*.

#attenzione()[
  L'inizializzazione dei parametri è fondamentale per il successo del training. Esiste un'intera branca di ricerca che studia come inizializzare i parametri: *il set di pesi di partenza influenza fortemente come il modello evolve* durante l'addestramento.
]

=== Strategie di Inizializzazione

*Inizializzazione casuale semplice*: *Non* consigliata
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

= Neural Networks: Regression vs Classification

Le reti neurali possono essere utilizzate per diversi tipi di predizioni: 
- *Regressione*: predizione di valori continui (es. prezzo di una casa)
- *Classificazione*: predizione di classi discrete (es. tipo di fiore)

#nota()[
  Anche se la struttura di base della rete, può sembrare simile, cambiano l'output layer e la funzione di loss:
   - Nella *regressione*, l'output layer è tipicamente un layer lineare senza attivazione. 
  - Nella *classificazione* l'output layer produce logits che vengono poi trasformati in probabilità tramite _softmax_. 
  
  La funzione di loss è *MSE* per la regressione e *Cross-Entropy* per la classificazione.
]

== Regressione

Nel problema di regressione, l'obbiettivo è *predirre* un valore continuo *$y in R$*, dato un input $x in R^D$. Il modello impara la seguente funzione:
$
  f(x) tilde y
$
Il layer di output è tipicamente un layer lineare senza attivazione:
$
  hat(y) = f(x;w;b)
$
Inoltre l'output $hat(y)$ del modello *non* ha bound, in quanto si tratta di un valore continuio. La funzione di loss più comune è la *Mean Squared Error* (MSE) o *Mean Absolute Error* (MAE).

== Classificazione

Dato un vettore di input $x in R^D$, il problema di classificazione consiste nel predire una classe *$y in {1,2,...,K}$*, dove $K$ è il numero di classi.

#nota()[
  Valgono le seguenti assunzioni: 
  - Le classi sono *disgiunte* e mutualmente esclusive (un esempio appartiene ad una sola classe)

  - Lo spazio dell'input viene partizionato in *regioni di decisione* (decision boundaries) che separano le classi.

  - I boundary tra le classi possono essere *lineari* o *non lineari* a seconda della complessità del modello e dei dati e prendono il nome di *decision surface*.
]

A loro volta i modelli di classificazione possono utilizzare due differenti approcci:
- *Funzioni Discriminative*: Assegnano direttamente una classe $y$ ad un input $x$ (es. SVM, Decision Tree)
- *Funzioni Discriminative probabilistiche*: Modellano la probabilità $P(C_k|x)$ e predicono la classe con la massima probabilità (es. Logistic Regression, Neural Networks)

=== Funzioni Discriminative
Una *funzione discriminativa* è un modello che mappa *direttamente* un input $x$ ad una classe $y$ senza modellare esplicitamente la distribuzione di probabilità. La funzione è così definita (caso *binario*):
$
  y(x): R^D -> R\
  y(x) = w^T x + b
$
Dove la regole di decisione è definita come:
- Se $y(x) > 0$, allora $x$ viene assegnato alla classe $C_1$.
- Se $y(x) <= 0$, allora $x$ viene assegnato alla classe $C_2$.

I modelli così definiti imparano a tracciare confini decisionali nello spazio degli input per separare le classi. Nel caso binario il *decision boundary* è definito da:
$
  y(x) = 0
$
Tale bound decisionale, corrisponde ad un iperpiano nello spazio degli input.

#esempio()[
  Consideriamo un classificatore lineare binario in 2D con:
  - Vettore dei pesi: $w = vec(1, 1)$
  - Bias: $b = -3$

  La funzione discriminativa è quindi:
  $
    y(x) = w^T x + b = x_1 + x_2 - 3
  $

  Il *decision boundary* è l'insieme di punti dove $y(x) = 0$:
  $
    x_1 + x_2 - 3 = 0 space arrow.r.double space x_2 = 3 - x_1
  $

  Questa equazione rappresenta una *retta* nello spazio 2D. I punti vengono classificati come:
  - $mb("Classe")$ $mb(C_1)$ se $y(x) > 0$ (regione blu: $x_1 + x_2 > 3$)
  - $mr("Classe")$ $mr(C_2)$ se $y(x) <= 0$ (regione rossa: $x_1 + x_2 <= 3$)

  #figure(
    cetz.canvas({
      import cetz.draw: *

      // Assi
      line((0, 0), (5, 0), mark: (end: ">"))
      content((5.3, 0), $x_1$)
      line((0, 0), (0, 5), mark: (end: ">"))
      content((0, 5.3), $x_2$)

      // Griglia leggera
      for i in range(1, 5) {
        line((i, 0), (i, 5), stroke: (paint: gray.lighten(70%), thickness: 0.5pt))
        line((0, i), (5, i), stroke: (paint: gray.lighten(70%), thickness: 0.5pt))
      }

      // Tacche
      for i in range(1, 5) {
        line((i, -0.1), (i, 0.1))
        content((i, -0.3), text(size: 8pt, str(i)))
        line((-0.1, i), (0.1, i))
        content((-0.3, i), text(size: 8pt, str(i)))
      }

      // Regione C1 (y(x) > 0) - sfondo blu chiaro
      let region1 = ((3, 0), (5, 0), (5, 5), (0, 5), (0, 3), (3, 0))
      line(..region1, fill: blue.lighten(85%), stroke: none)

      // Regione C2 (y(x) <= 0) - sfondo rosso chiaro
      let region2 = ((0, 0), (3, 0), (0, 3), (0, 0))
      line(..region2, fill: red.lighten(85%), stroke: none)

      // Decision boundary: x2 = 3 - x1
      line((0, 3), (3, 0), stroke: (paint: black, thickness: 2pt))
      content((3.3, 1.3), text(size: 9pt, $y(x) = 0$), anchor: "south-east")

      // Punti classe C1 (blu) - y(x) > 0
      circle((3.5, 2.5), radius: 0.1, fill: blue, stroke: none)
      circle((4, 1.5), radius: 0.1, fill: blue, stroke: none)
      circle((2.5, 3), radius: 0.1, fill: blue, stroke: none)
      circle((4.5, 3.5), radius: 0.1, fill: blue, stroke: none)
      circle((3, 2), radius: 0.1, fill: blue, stroke: none)

      // Punti classe C2 (rossi) - y(x) <= 0
      circle((0.5, 0.5), radius: 0.1, fill: red, stroke: none)
      circle((1.5, 0.8), radius: 0.1, fill: red, stroke: none)
      circle((0.8, 1.5), radius: 0.1, fill: red, stroke: none)
      circle((2, 0.5), radius: 0.1, fill: red, stroke: none)
      circle((1, 1), radius: 0.1, fill: red, stroke: none)

      // Etichette regioni
      content((4, 4.5), text(size: 10pt, $mb(C_1)$), fill: blue)
      content((0.7, 0.3), text(size: 10pt, $mr(C_2)$), fill: red)

      // Freccia vettore normale w
      line((1.5, 1.5), (2.2, 2.2), mark: (end: ">"), stroke: (paint: purple, thickness: 1.5pt))
      content((2.4, 2.4), text(size: 9pt, $w$), fill: purple)
    }),
    caption: [Decision boundary per un classificatore lineare binario. La retta $x_1 + x_2 = 3$ separa lo spazio in due regioni. Il vettore $w$ è perpendicolare al boundary e punta verso la regione $C_1$.],
  )

  #nota()[
    In spazi di dimensione superiore ($D > 2$), il decision boundary diventa un *iperpiano* di dimensione $D-1$. Il vettore dei pesi $w$ è sempre *perpendicolare* al boundary e definisce la direzione di massima variazione.
  ]
] 

*Proprietà* di queste funzioni: 
- Mapping diretto da input a classe
- Non modellano esplicitamente la probabilità
- Non forniscono una misura di confidenza nelle predizioni
- Possono essere più semplici da addestrare in alcuni casi, ma meno flessibili rispetto a modelli probabilistici

=== Funzioni Discriminative probabilistiche

Questo tipo di modelli, invece di assegnare direttamente una classe, modellano la probabilità *$P(C_k|x)$* per ogni classe $C_k$ dato l'input $x$. La predizione finale è la classe con la massima probabilità:
$
  hat(y) = arg max_k P(C_k | x)
$
I parametri del modello vengono appresi massimizzando la probabilità dei dati osservati (*massimizzazione della verosimiglianza*) o minimizzando la Cross-Entropy Loss.

Per quanto riguarda il layer di output, è necessario che produca dei *logits* (valori reali arbitrari) che vengono poi trasformati in probabilità tramite la funzione *Softmax*:
- *Caso binario*: viene utilizzata una funzione *Sigmoid* per mappare i logits in probabilità 
$
  p(C_1 | x ) = sigma(w^T x)
$
- *Caso multiclasse*: viene utilizzata la funzione *Softmax* per ottenere una distribuzione di probabilità su tutte le classi:
$
  P(C_k | x) = "softmax"(W x)_k 
$

#nota()[
  Questi modelli forniscono non solo una predizione di classe, ma anche una stima della *confidenza* in quella predizione (la probabilità associata). Questo è particolarmente utile in molte applicazioni reali dove è importante sapere quanto il modello è sicuro delle sue predizioni.
]

== Applicazione in Pytorch

=== Dataloader




=== Esempio: Regressione non lineare con MLP

=== Esempio: Classificazione multiclasse con MLP






