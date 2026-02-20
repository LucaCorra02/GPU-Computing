#import "../template.typ": *

= PyTorch

Si tratta di una libreria che permette di andare a costruire modelli di deep learning. La maggior parte della libreria è scritta in `C++/Cuda`.

== Tensori

I tensori sono l'*unità base* di PyTorch. Si tratta di una generalizzazione di una scalare, vettore o matrice (una sorta di _wrap_). Aggiunge una serie di operazioni in più.\
Un tensore può avere diverse dimensioni:
- $0D -> "Scalari"$
- $1D -> "Vettori"$
- $2D -> "Matrici"$
- $3D+ -> "Tensori a grande dimensioni"$

In particolare un tensore, contiene:
- *Dati*, valori che incapsula
- *Type* (`dtype`), i dati che contiene avranno un certo tipico. Il tipo determina: precisione, utilizzo di memoria, operazioni valide
- *Device*, se risiede su `CPU` o `GPU`. Possono quindi essere eseguiti sulla GPU e possono sfruttare l'accelerazione hardware

Vengono usati per modellare tutta la parte _nuemerica_ del nostro modello: input, output e iperparametri.

Per inizializzare un tensore da dei dati, possiamo:
```py
  import torch
  data = [[1,2],[3,4]]
  x_data = torch.tensor(data)
  print(x_data)
```

#note()[
  Il tipo viene dedotto automaticamente dall'interprete.
]

Inoltre è possibile costruire un tensore da un array `numpy` e viceversa:
```py
import numpy as np
data = [[1,2],[3,4]]
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)
```
Il vantaggio è che le prorpeità (`shape,datatype`) del vettore `numpy` vengono *ereditate* dal `tensor`, a meno che non vengano sovrascritte:
```py
x_ones = torch.ones_like(x_data) # [[1,1],[1,1]]
x_rand = torch.rand_like(x_data, dtype=torch.float)
#[[1.2,0.3],[0.23,2.3]]
```
#warning()[
  Quando si converte un vettore `numpy` ad un `tensor` ci sono due aspetti da considerare:
  - `PyTorch` di default usa i `float32`, mentre `numpy` usa i `float64`. Se ereditassimo direttamente da `numpy` ci potrebbe essere un errore di conversione, bisgona *riconvertire* corretamente i dati.

  - Un `numpy` array quando diventa un `tensor`, continua a *condividere la memoria* (puntatore alla stessa locazione di memoria). Se si effetuano dei cambiamenti sul `tensor`, influiscono anche sull'array `numpy`.
]

=== Shape

La dimensione di un `tensor` si leggono dall'esterno verso l'interno. Ad esempio:
```py
  tensor(
    [[
      [1,2,3],
      [4,5,6],
      [7,8,9]
    ]]
  )
  torch.size([1,3,3]) # dim 0,1,2
```
In questo caso è un tensore 3D, c'è una dimensione $3*3$.

== Operazioni

Di default le operazioni tra tensori possono essere eseguite sia sulla `CPU` che sulla `GPU`, generalmente i tensori vengono allocati inizialmente sulla `CPU`.

Esistono inoltre degli operatori caricati, ad esempio:
- `@`: utilizzato per *operazioni di algebra lineare*, come il prodotto matriciale
- `*`: utilizzato per *element-wise product*, ad esempio prodotto degli elementi cella per cella di un vettore

#warning()[
  La scrittura `tensor.add_(5)` serve per le operazioni *in place*. Esse modificano il tensore in maniera diretta.
]


=== Broadcasting

Si tratta di un'operazione fatta in automatico da `PyTorch` quando i *batch* (prime due dimensioni) di due tensori non corrispondono:
```py
a = torch.randn(2, 4, 5, 4) # [B1, B2, 5, 4]
b = torch.randn(2, 1, 4, 3) # [B1, 1, 4, 3]
c = torch.matmul(a, b)
print("Shape of c:", c.shape) # [2, 4, 5, 3]
```
Nell'esempio la dimensione dei batch di $a$ e $b$ non corrispondono, in particolare:
- $a "batch"(2,4)$
- $b "batch"(2,1)$
#figure(
  image("../assets/broadcasting.png", width: 65%),
  caption: [
    Rappresentazione dei tensori $a$ e $b$ (input)
  ],
)
Siccome le dimensioni dei batch non corrispondono viene fatto *broadcasting* (in modo implicito). Il tensore con dimensione minore viene espanso tante volte quanto serve per arrivare alla dimensione dell'operando più grande, in modo da croprire così il missmatch.\
Nell'esempio sorpa il la seconda dimensione del batch del tenosore $b$ viene posta a $4$. Il tensore $c$ risultante avrà una `c.shape == (2, 4, 5, 3)`, dove:
- $2 ->$ batch esterno
- $4 ->$ gruppo interno del batch
- $5 ->$ numero di righe per ogni matrice
- $3 ->$ numero di colonne per ogni matrice

#figure(
  image("../assets/broadcasting-result.png", width: 65%),
  caption: [
    Rappresentazione dei tensori $a$ e $b$ (input)
  ],
)

#warning()[
  Non tutte le forme di tensori sono compatibili. Affinché il broadcasting funzioni, `PyTorch` confronta le dimensioni dei due tensori partendo da *destra verso sinistra* (dall'ultima dimensione alla prima). Due dimensioni sono *compatibili* se:
  - Sono uguali
  - Una delle due è $1$.

  Se un tensore ha meno dimensioni dell'altro, `PyTorch` aggiunge virtualmente delle dimensioni $1$ a sinistra.
]

#example()[
  Supponiamo di avere il seguente codice:
  ```py
    A = torch.tensor([[1.], [2.], [3.], [4.]])
    B = torch.tensor([[5., -5., 5., -5., 5.]])
    C = A+B
  ```
  Dove $A$ è un vettore colonna $4 times 1$, mentre $B$ è un vettore riga $1 times 5$. In questo caso sia $A$ che $B$ vengono espansi ad una matrice $4 times 4$:

  #figure(
    scale(80%)[
      #grid(
        columns: (auto, auto, auto),
        column-gutter: 1.5em,
        row-gutter: 0.8em,
        align: horizon,

        // Colonna sinistra: A e B
        [
          #align(center)[
            #text(size: 8pt, weight: "bold", fill: blue)[*A*]
            #table(
              columns: 1,
              stroke: 0.5pt + black,
              align: center + horizon,
              inset: 3pt,
              [1],
              [2],
              [3],
              [4],
            )
          ]

          #v(0.8em)

          #align(center)[
            #text(size: 8pt, weight: "bold", fill: blue)[*B*]
            #table(
              columns: 5,
              stroke: 0.5pt + black,
              align: center + horizon,
              inset: 3pt,
              [5], [−5], [5], [−5], [5],
            )
          ]
        ],

        // Colonna centrale: frecce e replicazioni
        [
          #v(0.3em)
          #align(center)[
            #text(size: 7pt, fill: red, weight: "bold")[replicate]
            #text(size: 10pt, fill: red)[⟷]

            #v(0.2em)

            #table(
              columns: 5,
              stroke: 0.5pt + black,
              align: center + horizon,
              inset: 3pt,
              [1], [1], [1], [1], [1],
              [2], [2], [2], [2], [2],
              [3], [3], [3], [3], [3],
              [4], [4], [4], [4], [4],
            )
          ]

          #v(0.8em)

          #align(center)[
            #text(size: 7pt, fill: red, weight: "bold")[replicate]
            #text(size: 10pt, fill: red)[⟺]

            #v(0.2em)

            #table(
              columns: 5,
              stroke: 0.5pt + black,
              align: center + horizon,
              inset: 3pt,
              [5], [−5], [5], [−5], [5],
              [5], [−5], [5], [−5], [5],
              [5], [−5], [5], [−5], [5],
              [5], [−5], [5], [−5], [5],
            )
          ]
        ],

        // Colonna destra: frecce e risultato
        [
          #v(1.5em)
          #align(center)[
            #text(size: 12pt)[↘]

            #v(0.3em)

            #table(
              columns: 5,
              stroke: 0.5pt + black,
              align: center + horizon,
              inset: 3pt,
              [6], [−4], [6], [−4], [6],
              [7], [−3], [7], [−3], [7],
              [8], [−2], [8], [−2], [8],
              [9], [−1], [9], [−1], [9],
            )

            #v(0.3em)
            #text(size: 9pt, weight: "bold", fill: blue)[$C = A + B$]

            #v(1.5em)
            #text(size: 14pt)[↗]
          ]
        ],
      )
    ],
    caption: [Broadcasting nella somma],
  )
]


=== Torch Einsum

Si tratta di una notazione per effetuare in modo coinciso operazioni compoment-wise. `torch.einsum` prende come argomento una stringa che descrive:
- l'operazione che si vuole effettuare
- i tensori su cui si deve operare
- il tensore risultato

#note()[
  L'operazione viene eseguita su tutti gli indici che non appaiono tra gli indici del risultato.
]

*Prodotto tra matrici*: Date due matrici $P "e" Q$ il prodotto:
$
  M_(i,k) = sum_j P_(i,j)Q_(j,k)
$
può essere scritta come `torch.einsum('ij,jk->ik', P, Q)`

*Prodotto matrice vettore*: Dato un vettore $M$ e $v$:
$
  w_i = sum_j M_(i,j)v_j
$
può essere scritta come `torch.einsum("ij,j->i", M, v)`

*Prodotto element-wise*: Date due matrici $P "e" Q$:
$
  M_(i,j) = P_(i,j)*Q_(i,j)
$
può essere scritta come `torch.einsum("ij,ij->ij", P, Q)`

*Prodotto Matriciale a batch*: Dati due tensori $3D$ $P "e" Q$:
$
  M_(n,i,k) = sum_j (P_(n,i,j)Q_(n,j,k))
$
può essere scritta come `torch.einsum("nij,njk->nik", P, Q)`

=== Media e Deviazione

Calcolare la media e la deviazione standard sui tensori è molto comune. In particolare, vengono utilizzate per eseguire data normalization e batch normalization nei modelli di deep learning. Le funzioni sono due:
- `torch.mean()`
- `torch.std()`
In entrambe le funzioni è possibile specificare su quali dimensioni del tensore non si vuole fare la media.

#table(
  columns: 5,
  [_Input_], [_Shape_], [_Media_], [_STD_], [_Risultato_],
  [Vettore $1D$], [`[N]`], [`x.mean()`], [`x.std()`], [Scalare],
  [Matrice $2D$], [`[M,N]`], [`x.mean(dim=0 or dim=1)`], [`x.std(dim=0 or dim=1)`], [Media per riga o colonna],
  [Batch immagine], [`[B,3,H,W]`], [`x.mean(0,2,3)`], [`x.std(0,2,3)`], [Media per canale RGB],
)

== Normalizzazione dei batch

Nei modelli di deep learning, la normalizzazione dei batch ha un ruolo molto fondamentale. Tale tecnina viene usata per rendere l'*addestramento* delle reti neurali *più veloce* e *più stabile*.

Tale tecnica, va a risolvere il problema del $mr("internal covariate shift")$. Quando si va ad adestrare una rete neurale, i pesi del primo layer cambiano man mano, cambiando la distribuzione dei dati che arriva al secondo layer. Il secondo layer deve quindi _riadattarsi_ continuamente alla nuova distribuzione, rallentando l'apprendimento.\

La soluzione prende il nome di $mg("batch norm")$. Essa va a  normalizzare l'input di ogni layer nascosto per stabilizzare la distribuzione. Per ogni mini-batch di dati durante il training, il layer BatchNorm esegue $3$ passaggi:

- Calcola la media $mu$ e la varianza $sigma^2$ del batch corrente.

- *Normalizzazione*: Sottrae la media e divide per la deviazione standard (aggiungendo un numero piccolo $epsilon$ per evitare le divisione per zero):
  $
    hat(x) = (x - mu)/(sqrt(sigma^2+epsilon))
  $
  I dati del batch avranno ora media $0$ e varianza $1$

- *Scale and shift*: I dati normalizzati vengono moltiplicati per un iper-parametro $gamma$ e un parametro $beta$ appreso dal modello:
  $
    y = gamma hat(x)+ beta
  $
  #note()[
    Se venisse forzata media $0$ e varianza $1$, potremmo limitare la capacità della rete. $gamma$ e $beta$ permettono alla rete di imparare se le serve una distribuzione diversa (o addirittura di annullare la normalizzazione se necessario).
  ]

#informally()[
  I vantaggi introdotti sono due:
  - *Velocità*: Permette di usare Learning Rate molto più alti (convergenza rapida).

  - *Stabilità*: Rende la rete meno sensibile all'inizializzazione casuale dei pesi.

  - *Regolarizzazione*: Introduce un leggero "rumore" (poiché le statistiche dipendono dal batch casuale), agendo come un leggero Dropout ed evitando l'overfitting.
]

=== batchNorm2d

In Pytorch `nn.BatchNorm2d` è l'implementazione del layer batch norm specifica per dati $4D$ (immagini), tipicamente usata nelle Reti Neurali Convoluzionali (CNN).

Il layer prende in input un tensore con la seguente shape $(B, C, H, W)$ dove:
- $B$ è la batch size
- $C$ è il numero di canali
- $H,W$ sono le dimensioni spaziali

Il layer esegue una *channel-wise batch normalization*. Per ogni canale $c$ calcola la media e la varianza su tutte le altre dimensioni ($c$ rimane fisso):
$
  mu_c = 1 / ("BHW") sum_(b,h,w) x_(b,c,h,w) \
  sigma^2_c = 1 / ("BHW") sum_(b,h,w) (x_(b,c,h,w) - mu_c)^2
$

#note()[
  I canali $C$ rappresentano le features estratte (embedding). La BatchNorm normalizza ogni feature indipendentemente dalle altre, ma coerentemente su tutto il batch $B$ e su tutta l'immagine spaziale.
]

Successivamente viene applicata la normalizzazione al batch:
$
  hat(x)_(b,c,h,w) = (x_(b,c,h,w)-mu_c)/(sqrt(sigma^2_c + epsilon))
$

== Tensor Reshaping

Quando viene creato un tensore, i dati vengono memorizati in *maniera lineare* in memoria. Le funzioni `view` e `reshape` cambiano la dimensione dei dati *senza copiarli* (viene cambiata solamente la loro _visualizzazione_).

=== Transpose

L'operazione `transpose()` o `.t()` *scambia due dimensioni* di un tensore. Per matrici 2D, scambia righe e colonne.

#figure(
  cetz.canvas({
    import cetz.draw: *

    // Primo esempio: 2x3 -> 3x2
    content((0, 3), [#text(10pt)[`x = torch.tensor([[1, 3, 0], [2, 4, 6]])`]])

    // Matrice originale 2x3
    let colors = (green, yellow, red, blue, fuchsia, eastern)
    let idx = 0
    for i in range(2) {
      for j in range(3) {
        rect((j * 0.8, -i * 0.8 + 2), (j * 0.8 + 0.8, -i * 0.8 + 2 - 0.8), fill: colors.at(idx), stroke: black)
        idx += 1
      }
    }

    // Freccia
    content((3, 1.2), text(20pt)[→])

    // Matrice trasposta 3x2
    content((5.5, 3), [#text(10pt)[`x.t()`]])
    let data = ((0, 3), (2, 4), (1, 5))
    idx = 0
    for i in range(3) {
      for j in range(2) {
        let color_idx = data.at(i).at(j)
        rect(
          (j * 0.8 + 4.5, -i * 0.8 + 2),
          (j * 0.8 + 4.5 + 0.8, -i * 0.8 + 2 - 0.8),
          fill: colors.at(color_idx),
          stroke: black,
        )
      }
    }
  }),
  caption: [Transpose: scambio tra righe e colonne],
)

==== Transpose su tensori multidimensionali

Per tensori con più di 2 dimensioni, `transpose(dim0, dim1)` scambia le dimensioni specificate. Questo è utile per riorganizzare i dati mantenendo le altre dimensioni invariate.

#figure(
  cetz.canvas({
    import cetz.draw: *

    // Tensore originale 4D: shape (2, 2, 3, 1) visualizzato come matrici sovrapposte
    content((0, 5), [#text(9pt)[`x = torch.tensor([[[[1, 2, 1],`]])
    content((0, 4.6), [#text(9pt)[`                  [[2, 1, 2]],`]])
    content((0, 4.2), [#text(9pt)[`                 [[[3, 0, 3],`]])
    content((0, 3.8), [#text(9pt)[`                  [[0, 3, 0]]]])`]])

    // Primo livello (layer 0)
    let colors1 = (green, blue, green, yellow, red, yellow)
    let data1 = (1, 2, 1, 2, 1, 2)
    for i in range(2) {
      for j in range(3) {
        rect(
          (j * 0.5, -i * 0.5 + 3),
          (j * 0.5 + 0.5, -i * 0.5 + 3 - 0.5),
          fill: colors1.at(i * 3 + j),
          stroke: black + 1pt,
        )
      }
    }

    // Secondo livello (layer 1) - leggermente offset per effetto 3D
    let colors2 = (green, eastern, green, eastern, green, eastern)
    for i in range(2) {
      for j in range(3) {
        rect(
          (j * 0.5 + 0.15, -i * 0.5 + 3 - 0.15),
          (j * 0.5 + 0.5 + 0.15, -i * 0.5 + 3 - 0.5 - 0.15),
          fill: colors2.at(i * 3 + j),
          stroke: black + 1pt,
        )
      }
    }

    // transpose(0, 1) - scambia dimensioni 0 e 1
    content((2.5, 2.3), text(14pt)[→])
    content((4.5, 5), [#text(9pt)[`x.transpose(0, 1)`]])

    // Risultato transpose(0, 1)
    let colors1_t01 = (green, green, blue, eastern, green, green)
    for i in range(2) {
      for j in range(3) {
        rect(
          (j * 0.5 + 3.5, -i * 0.5 + 3),
          (j * 0.5 + 3.5 + 0.5, -i * 0.5 + 3 - 0.5),
          fill: colors1_t01.at(i * 3 + j),
          stroke: black + 1pt,
        )
      }
    }

    let colors2_t01 = (yellow, eastern, red, green, yellow, eastern)
    for i in range(2) {
      for j in range(3) {
        rect(
          (j * 0.5 + 3.5 + 0.15, -i * 0.5 + 3 - 0.15),
          (j * 0.5 + 3.5 + 0.5 + 0.15, -i * 0.5 + 3 - 0.5 - 0.15),
          fill: colors2_t01.at(i * 3 + j),
          stroke: black + 1pt,
        )
      }
    }

    // transpose(0, 2) - scambia dimensioni 0 e 2
    content((2.5, 0.8), text(14pt)[→])
    content((4.5, 1.5), [#text(9pt)[`x.transpose(0, 2)`]])

    // Risultato transpose(0, 2)
    let colors1_t02 = (green, yellow, blue, red, green, yellow)
    for i in range(2) {
      for j in range(3) {
        rect(
          (j * 0.5 + 3.5, -i * 0.5 + 1.2),
          (j * 0.5 + 3.5 + 0.5, -i * 0.5 + 1.2 - 0.5),
          fill: colors1_t02.at(i * 3 + j),
          stroke: black + 1pt,
        )
      }
    }

    let colors2_t02 = (green, eastern, green, eastern, green, eastern)
    for i in range(2) {
      for j in range(3) {
        rect(
          (j * 0.5 + 3.5 + 0.15, -i * 0.5 + 1.2 - 0.15),
          (j * 0.5 + 3.5 + 0.5 + 0.15, -i * 0.5 + 1.2 - 0.5 - 0.15),
          fill: colors2_t02.at(i * 3 + j),
          stroke: black + 1pt,
        )
      }
    }
  }),
  caption: [Transpose su tensori 4D: scambio di dimensioni specifiche],
)

#informally()[
  - `transpose(0, 1)`: scambia le *prime due dimensioni* (es. batch e canali). Utile quando si vuole riorganizzare l'ordine dei batch rispetto ai canali.
  - `transpose(0, 2)`: scambia la *prima e terza dimensione* (es. batch e altezza). Permette di riordinare i dati spaziali rispetto al batch.

  La trasposizione è fondamentale in operazioni come la convoluzione trasposta o quando si passano dati tra layer con formati diversi (es. NCHW ↔ NHWC).
]

#note()[
  La trasposizione *non copia i dati* in memoria, ma cambia solo l'interpretazione degli indici. Questo può rendere il tensore *non contiguo* in memoria.
]

=== View

Esempio di utilizzo di `view`:
```py
  x.arrange(12) # x è un tentore da 0 a 12
  x.view(3,4) # stessi dati di x riorganizzati in 3*4
  tensor([[0,1,2,3],
          [4,5,6,7],
          [8,9,10,11]])
```

#figure(
  cetz.canvas({
    import cetz.draw: *

    // Esempio 1: view(-1) - flattening
    content((0, 3.5), [#text(10pt)[`x = torch.tensor([[1, 3, 0], [2, 4, 6]])`]])

    // Matrice originale 2x3
    let colors = (green, yellow, red, blue, fuchsia, eastern)
    let idx = 0
    for i in range(2) {
      for j in range(3) {
        rect((j * 0.6, -i * 0.6 + 3), (j * 0.6 + 0.6, -i * 0.6 + 3 - 0.6), fill: colors.at(idx), stroke: black)
        idx += 1
      }
    }

    // Freccia
    content((2.5, 2.4), text(16pt)[→])

    // Vettore 1D
    content((5.5, 3.5), [#text(10pt)[`x.view(-1)`]])
    for i in range(6) {
      rect((i * 0.6 + 3.5, 3), (i * 0.6 + 3.5 + 0.6, 2.4), fill: colors.at(i), stroke: black)
    }

    // Esempio 2: view(3, -1) - rimodellamento
    content((0, 1.5), [#text(10pt)[`x = torch.tensor([[1, 3, 0], [2, 4, 6]])`]])

    // Matrice originale 2x3
    idx = 0
    for i in range(2) {
      for j in range(3) {
        rect((j * 0.6, -i * 0.6 + 1), (j * 0.6 + 0.6, -i * 0.6 + 1 - 0.6), fill: colors.at(idx), stroke: black)
        idx += 1
      }
    }

    // Freccia
    content((2.5, 0.4), text(16pt)[→])

    // Nuova forma 3x2
    content((5.5, 1.5), [#text(10pt)[`x.view(3, -1)`]])
    idx = 0
    for i in range(3) {
      for j in range(2) {
        rect(
          (j * 0.6 + 3.5, -i * 0.6 + 1),
          (j * 0.6 + 3.5 + 0.6, -i * 0.6 + 1 - 0.6),
          fill: colors.at(idx),
          stroke: black,
        )
        idx += 1
      }
    }
  }),
  caption: [View: riorganizzazione dei dati senza copia],
)

#warning()[
  `View` effettua il reshape solamente di tensori contigui in memoria.
  ```py
    x = torch.arange(12).view(3,4)
    y = x.trnspose(0,1)
    z = y.view(-1) #inferisce lui le dimensioni
  ```
  La matrice viene trasposta (scambio righe colonne) ed è per quello che non trova più la contiguità.

  #informally()[
    Il concetto di contiguità significa: _I numeri che sono vicini nella matrice (logica) sono vicini anche nella striscia di memoria RAM (fisica)_.
  ]

  - Matrice originale ($3 times 4$): La prima riga è `0, 1, 2, 3`, i dati in memoria sono contigui.
  - Matrice trasposta ($4 times 3$): La prima riga ora è composta dai numeri `0, 4, 8`. Tuttavia, la rappresentazione _logica_ non corrisponde con quella _fisica_. Ad esempio, lo $0$ e il $4$ non sono contigui in memoria, ci sono di mezzo `1, 2, 3` che appartengono ad altre righe della nuova matrice.

  `view(-1)` da errore in quanto la funzione restituisce i dati in fila, dall'inizio alla fine della memoria. Essendo eseguito sulla matrice trasposta, `view` andrebbe a leggere la memoria fisica nell'ordine originale: `0, 1, 2, 3....`. Tuttavia la matrice trasposta _logicamente_ dovrebbe iniziare con `0, 4, 8....`. C'è quindi un disaccordo tra l'ordine fisico e quello logico.

  La soluzione è usare `reshape` in questo caso.
]

=== Reshape

`reshape` è molto più flessibile. Tuttavia, può *implicare una ricopiatura dei dati se necessario*.

```py
  x = torch.arange(12).view(3, 4)
  y = x.t() # transpose -> non-contiguous
  z = y.reshape(-1) # works; makes a copy if needed
  print(z)
  tensor([0,4,8,1,5,9,2,6,10,3,7,11])
```

=== Expand

L'operazione `expand()` *replica i dati* lungo le dimensioni specificate *senza allocare memoria aggiuntiva* (usa broadcasting).

#figure(
  cetz.canvas({
    import cetz.draw: *

    content((0, 3.5), [#text(10pt)[`x = torch.tensor([[1, 3, 0], [2, 4, 6]])`]])

    // Matrice originale 2x3
    let colors = (green, yellow, red, blue, fuchsia, eastern)
    let idx = 0
    for i in range(2) {
      for j in range(3) {
        rect((j * 0.5, -i * 0.5 + 3), (j * 0.5 + 0.5, -i * 0.5 + 3 - 0.5), fill: colors.at(idx), stroke: black)
        idx += 1
      }
    }

    // Freccia
    content((2.2, 2.6), text(16pt)[→])

    // Espansione con .view(1, 2, 3).expand(3, 2, 3)
    content((6.5, 3.5), [#text(10pt)[`x.view(1, 2, 3).expand(3, 2, 3)`]])

    for d in range(3) {
      let offset = d * 0.3
      idx = 0
      for i in range(2) {
        for j in range(3) {
          let alpha = if d == 0 { 100% } else { 70% }
          rect(
            (j * 0.5 + 3.5 + offset, -i * 0.5 + 3 - offset),
            (j * 0.5 + 3.5 + 0.5 + offset, -i * 0.5 + 3 - 0.5 - offset),
            fill: colors.at(idx).lighten(d * 20%),
            stroke: black,
          )
          idx += 1
        }
      }
    }
  }),
  caption: [Expand: replicazione dei dati mediante broadcasting],
)

#note()[
  `expand()` *non alloca nuova memoria* ma semplicemente modifica gli stride del tensore per far sì che gli stessi dati vengano "visti" più volte. Questa operazione è molto efficiente rispetto a una vera copia dei dati.

  *Differenze chiave*:
  - `view()` / `reshape()`: cambiano la _forma_ dei dati esistenti
  - `transpose()`: scambia _dimensioni_
  - `expand()`: _replica virtualmente_ i dati lungo nuove dimensioni
]

=== Add/Remove dim

Per tali scopi esistono due funzioni:
- *`unsqueexe(dim)`* = aggiunge una nuova dimensione di size $1$ alla posizione dim. Da tensore $1D$ a tensore $2D$:
  ```py
  x = torch.tensor([1, 2, 3, 4])# shape([4])
  x_unsqueezed = x.unsqueeze(0)# shape([1,4])
  ```

- *`squeeze(dim)`* = operazione contraria, rimuove degli $1$ dalle dimensioni specificate:
  ```py
    y = torch.zeros(1, 3, 1, 5)# Size([1,3,1,5])
    y_squeezed = y.squeeze()# Size([3,5])

  ```
=== Concatenation / Splitting

Le operazioni di concatenazione e splitting permettono di combinare o dividere tensori lungo dimensioni specifiche.

==== torch.cat

Concatena una sequenza di tensori *lungo una dimensione esistente*. Tutti i tensori devono avere la stessa forma eccetto nella dimensione di concatenazione.

```py
x1 = torch.tensor([[1, 2], [3, 4]])  # Shape: [2, 2]
x2 = torch.tensor([[5, 6], [7, 8]])  # Shape: [2, 2]

# Concatenazione lungo dim=0 (righe)
result = torch.cat([x1, x2], dim=0)
# tensor([[1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]])  # Shape: [4, 2]

# Concatenazione lungo dim=1 (colonne)
result = torch.cat([x1, x2], dim=1)
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])  # Shape: [2, 4]
```

#figure(
  cetz.canvas({
    import cetz.draw: *

    // Esempio cat dim=0
    content((0, 3.5), [#text(9pt)[`torch.cat([x1, x2], dim=0)`]])

    // x1
    rect((0, 3), (0.6, 2.4), fill: green, stroke: black)
    rect((0.6, 3), (1.2, 2.4), fill: yellow, stroke: black)
    rect((0, 2.4), (0.6, 1.8), fill: blue, stroke: black)
    rect((0.6, 2.4), (1.2, 1.8), fill: red, stroke: black)

    content((1.6, 2.4), text(16pt)[+])

    // x2
    rect((2, 3), (2.6, 2.4), fill: fuchsia, stroke: black)
    rect((2.6, 3), (3.2, 2.4), fill: orange, stroke: black)
    rect((2, 2.4), (2.6, 1.8), fill: purple, stroke: black)
    rect((2.6, 2.4), (3.2, 1.8), fill: eastern, stroke: black)

    content((3.6, 2.4), text(16pt)[=])

    // Risultato
    rect((4, 3), (4.6, 2.4), fill: green, stroke: black)
    rect((4.6, 3), (5.2, 2.4), fill: yellow, stroke: black)
    rect((4, 2.4), (4.6, 1.8), fill: blue, stroke: black)
    rect((4.6, 2.4), (5.2, 1.8), fill: red, stroke: black)
    rect((4, 1.8), (4.6, 1.2), fill: fuchsia, stroke: black)
    rect((4.6, 1.8), (5.2, 1.2), fill: orange, stroke: black)
    rect((4, 1.2), (4.6, 0.6), fill: purple, stroke: black)
    rect((4.6, 1.2), (5.2, 0.6), fill: eastern, stroke: black)

    // Esempio cat dim=1
    content((0, 0), [#text(9pt)[`torch.cat([x1, x2], dim=1)`]])

    // x1
    rect((0, -0.5), (0.6, -1.1), fill: green, stroke: black)
    rect((0.6, -0.5), (1.2, -1.1), fill: yellow, stroke: black)
    rect((0, -1.1), (0.6, -1.7), fill: blue, stroke: black)
    rect((0.6, -1.1), (1.2, -1.7), fill: red, stroke: black)

    content((1.6, -1.1), text(16pt)[+])

    // x2
    rect((2, -0.5), (2.6, -1.1), fill: fuchsia, stroke: black)
    rect((2.6, -0.5), (3.2, -1.1), fill: orange, stroke: black)
    rect((2, -1.1), (2.6, -1.7), fill: purple, stroke: black)
    rect((2.6, -1.1), (3.2, -1.7), fill: eastern, stroke: black)

    content((3.6, -1.1), text(16pt)[=])

    // Risultato
    rect((4, -0.5), (4.6, -1.1), fill: green, stroke: black)
    rect((4.6, -0.5), (5.2, -1.1), fill: yellow, stroke: black)
    rect((5.2, -0.5), (5.8, -1.1), fill: fuchsia, stroke: black)
    rect((5.8, -0.5), (6.4, -1.1), fill: orange, stroke: black)
    rect((4, -1.1), (4.6, -1.7), fill: blue, stroke: black)
    rect((4.6, -1.1), (5.2, -1.7), fill: red, stroke: black)
    rect((5.2, -1.1), (5.8, -1.7), fill: purple, stroke: black)
    rect((5.8, -1.1), (6.4, -1.7), fill: eastern, stroke: black)
  }),
  caption: [torch.cat: concatenazione lungo dimensioni esistenti],
)

==== torch.stack

Concatena una sequenza di tensori *creando una nuova dimensione*. Tutti i tensori devono avere esattamente la stessa forma.

```py
x1 = torch.tensor([[1, 2], [3, 4]])  # Shape: [2, 2]
x2 = torch.tensor([[5, 6], [7, 8]])  # Shape: [2, 2]

# Stacking lungo dim=0 (crea una nuova dimensione)
result = torch.stack([x1, x2], dim=0)
# Shape: [2, 2, 2]
# tensor([[[1, 2],
#          [3, 4]],
#         [[5, 6],
#          [7, 8]]])

# Stacking lungo dim=1
result = torch.stack([x1, x2], dim=1)
# Shape: [2, 2, 2]
```

#figure(
  cetz.canvas({
    import cetz.draw: *

    content((0, 3), [#text(9pt)[`torch.stack([x1, x2], dim=0)`]])

    // x1 e x2 come layer separati
    content((0, 2.5), [#text(8pt)[x1]])
    rect((0, 2.2), (0.6, 1.6), fill: green, stroke: black)
    rect((0.6, 2.2), (1.2, 1.6), fill: yellow, stroke: black)
    rect((0, 1.6), (0.6, 1), fill: blue, stroke: black)
    rect((0.6, 1.6), (1.2, 1), fill: red, stroke: black)

    content((1.6, 1.6), text(16pt)[→])

    content((2.5, 2.5), [#text(8pt)[x2]])
    rect((2.5, 2.2), (3.1, 1.6), fill: fuchsia, stroke: black)
    rect((3.1, 2.2), (3.7, 1.6), fill: orange, stroke: black)
    rect((2.5, 1.6), (3.1, 1), fill: purple, stroke: black)
    rect((3.1, 1.6), (3.7, 1), fill: eastern, stroke: black)

    content((4.2, 1.6), text(16pt)[→])

    // Risultato: due layer impilati
    content((5.5, 2.5), [#text(8pt)[result (3D)]])
    // Layer 1 (x1)
    rect((5, 2.2), (5.6, 1.6), fill: green, stroke: black + 1.5pt)
    rect((5.6, 2.2), (6.2, 1.6), fill: yellow, stroke: black + 1.5pt)
    rect((5, 1.6), (5.6, 1), fill: blue, stroke: black + 1.5pt)
    rect((5.6, 1.6), (6.2, 1), fill: red, stroke: black + 1.5pt)

    // Layer 2 (x2) - offset per effetto 3D
    rect((5.2, 2), (5.8, 1.4), fill: fuchsia, stroke: black + 1.5pt)
    rect((5.8, 2), (6.4, 1.4), fill: orange, stroke: black + 1.5pt)
    rect((5.2, 1.4), (5.8, 0.8), fill: purple, stroke: black + 1.5pt)
    rect((5.8, 1.4), (6.4, 0.8), fill: eastern, stroke: black + 1.5pt)
  }),
  caption: [torch.stack: crea una nuova dimensione impilando i tensori],
)

#note()[
  *Differenza tra `cat` e `stack`*:
  - `torch.cat`: unisce tensori lungo una dimensione *esistente*, aumentando la size di quella dimensione
  - `torch.stack`: crea una *nuova dimensione* e posiziona i tensori lungo quella dimensione
]

==== torch.chunk

Divide un tensore in un *numero specifico di chunk* (parti) lungo una dimensione. Se la divisione non è uniforme, l'ultimo chunk sarà più piccolo.

```py
x = torch.arange(12).view(4, 3)
# tensor([[ 0,  1,  2],
#         [ 3,  4,  5],
#         [ 6,  7,  8],
#         [ 9, 10, 11]])

# Divide in 2 chunk lungo dim=0
chunks = torch.chunk(x, chunks=2, dim=0)
# (tensor([[0, 1, 2],
#          [3, 4, 5]]),
#  tensor([[ 6,  7,  8],
#          [ 9, 10, 11]]))

# Divide in 3 chunk lungo dim=1
chunks = torch.chunk(x, chunks=3, dim=1)
# Ogni chunk ha shape [4, 1]
```

==== torch.split

Divide un tensore in chunk di *dimensioni specificate*. Più flessibile di `chunk` perché permette di specificare la dimensione esatta di ogni parte.

```py
x = torch.arange(12).view(4, 3)

# Split con dimensione uniforme
splits = torch.split(x, split_size_or_sections=2, dim=0)
# (tensor([[0, 1, 2],
#          [3, 4, 5]]),
#  tensor([[ 6,  7,  8],
#          [ 9, 10, 11]]))

# Split con dimensioni diverse
splits = torch.split(x, split_size_or_sections=[1, 2, 1], dim=0)
# (tensor([[0, 1, 2]]),          # size 1
#  tensor([[3, 4, 5],            # size 2
#          [6, 7, 8]]),
#  tensor([[9, 10, 11]]))        # size 1
```

#figure(
  cetz.canvas({
    import cetz.draw: *

    content((0, 3), [#text(9pt)[`torch.split(x, [1, 2, 1], dim=0)`]])

    // Tensore originale 4x3
    let colors = (green, yellow, red, blue, fuchsia, eastern, orange, purple, gray, maroon, olive, navy)
    let idx = 0
    for i in range(4) {
      for j in range(3) {
        rect((j * 0.5, -i * 0.5 + 2), (j * 0.5 + 0.5, -i * 0.5 + 2 - 0.5), fill: colors.at(idx), stroke: black)
        idx += 1
      }
    }

    content((2, 0.75), text(16pt)[→])

    // Risultati dello split
    content((3, 3), [#text(8pt)[chunk 1]])
    for j in range(3) {
      rect((j * 0.5 + 3, 2), (j * 0.5 + 3 + 0.5, 1.5), fill: colors.at(j), stroke: black)
    }

    content((5, 3), [#text(8pt)[chunk 2]])
    idx = 3
    for i in range(2) {
      for j in range(3) {
        rect((j * 0.5 + 5, -i * 0.5 + 2), (j * 0.5 + 5 + 0.5, -i * 0.5 + 2 - 0.5), fill: colors.at(idx), stroke: black)
        idx += 1
      }
    }

    content((7, 3), [#text(8pt)[chunk 3]])
    idx = 9
    for j in range(3) {
      rect((j * 0.5 + 7, 2), (j * 0.5 + 7 + 0.5, 1.5), fill: colors.at(idx), stroke: black)
      idx += 1
    }
  }),
  caption: [torch.split: divide il tensore in parti di dimensioni specificate],
)

#informally()[
  *Quando usare chunk vs split*:
  - `chunk`: quando vuoi dividere in un *numero fisso di parti* (es. dividere un batch in 4 parti uguali)
  - `split`: quando vuoi *controllare la dimensione* di ogni parte (es. train/validation split con proporzioni specifiche)
]


