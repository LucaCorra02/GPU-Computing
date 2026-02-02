#import "../template.typ": *

== Parallel reduction

La *reduction* è un operazione che va a ridurre con un operazione associativa (in questo caso la somma) gli elementi di un array in un solo elemento:
$
  (x_1, dots, x_n) -> s = sum_(i=1)^(n) x_i
$
#nota()[
  La somma può essere sostituita da altre operazioni associative come il prodotto, il massimo, il minimo ecc.
]

== Somma di array parallela

Sequenzialmente il problema ha una complessità pari a $O(n)$, dove $n$ è la dimensione dell'array.

Un approccio parallelo potrebbe sfruttare le seguenti idee:
- Ad ogni passo, metà degli elementi vengono sommati in parallelo.
- Il numero di thread attivi vengono dimezzati ad ogni passo.
- Occorre sincronizzare il lavoro dei thread ad ogni passo.

#attenzione()[
  L'array di input deve avere una dimensione pari ad una potenza di $2$ (es. $2^k$). In caso contrario, occorre inserire degli elementi di *padding* (zero) fino a raggiungere la dimensione successiva pari ad una potenza di $2$.
]

=== Versione con divergenza

In una prima versione potremmo dividere i thread in due _gruppi_, attraverso il loro ID. In particolare, i thread con ID pari eseguono la somma del loro elemento con quello del thread successivo (ID dispari).

Ad ogni passo *lavora sola la metà dei thread* (con ID pari), rispetto al numero di elementi ancora da sommare.

```Python
@cuda.jit
def blockParReduce(array, out): # out ha dim = num_blocchi
  tid = cuda.threadIdx.x
  idx = cuda.grid(1) # indice globale
  n = len(array)
  if idx >= n: return

  # offset per il blocco
  block_skip = cuda.blockIdx.x * cuda.blockDim.x

  stride = 1
  while stride < cuda.blockDim.x:
      if (tid % (2 * stride)) == 0:
          array[block_skip + tid] += array[block_skip + tid + stride]
      cuda.syncthreads()
      stride *= 2

  # Risultato parziale del blocco
  if tid == 0:
      out[cuda.blockIdx.x] = array[block_skip]
```

// TODO: if idx >= n: return possibile deadlock?

#nota()[
  Serve *sincronizzazione* tra uno step e il successivo. Ogni step richiede che i risultati del passo precedente siano stati elaborati.

  Inoltre, è l'host che si occuperà di riunire i risultati parziali di ogni blocco (fuori dal kernel).
]

Graficamente il funzionamento è il seguente (supponendo che un blocco abbia si occupi di $8$ elementi dell'array originale):

#figure(
  {
    import cetz.draw: *

    cetz.canvas({
      let cell_w = 0.5
      let cell_h = 0.5
      let row_gap = 1.8

      // Array iniziale
      let values0 = (1, 3, 5, 7, 2, 4, 6, 8)
      for i in range(8) {
        let x = i * cell_w
        rect((x, 0), (x + cell_w, cell_h), fill: rgb(240, 240, 240), stroke: black)
        content((x + cell_w / 2, cell_h / 2), text(size: 8pt, weight: "bold")[#values0.at(i)])
      }

      // Thread ID step 1 (4 thread: 0,1,2,3)
      for i in range(4) {
        let x = i * cell_w * 2 + cell_w / 2
        circle((x, -0.6), radius: 0.18, fill: rgb(255, 150, 50), stroke: black)
        content((x, -0.6), text(size: 7pt, fill: white, weight: "bold")[#i])

        // Frecce che connettono thread ai due elementi da sommare
        line((x, -0.4), (x, -0.1), stroke: (paint: rgb(255, 100, 0), thickness: 1pt))
        line((x, -0.4), (x + cell_w, -0.1), stroke: (paint: rgb(255, 100, 0), thickness: 1pt))
      }

      // Array dopo step 1
      let values1 = (4, 3, 12, 7, 6, 4, 14, 8)
      let y1 = -row_gap
      for i in range(8) {
        let x = i * cell_w
        let fill_color = if calc.rem(i, 2) == 0 { rgb(220, 200, 250) } else { rgb(240, 240, 240) }
        rect((x, y1), (x + cell_w, y1 + cell_h), fill: fill_color, stroke: black)
        content((x + cell_w / 2, y1 + cell_h / 2), text(size: 8pt, weight: "bold")[#values1.at(i)])
      }

      // Thread ID step 2 (2 thread: 0,1)
      for i in range(2) {
        let x = i * cell_w * 4 + cell_w / 2
        circle((x, y1 - 0.6), radius: 0.18, fill: rgb(255, 150, 50), stroke: black)
        content((x, y1 - 0.6), text(size: 7pt, fill: white, weight: "bold")[#i])

        // Frecce
        line((x, y1 - 0.4), (x, y1 - 0.1), stroke: (paint: rgb(255, 100, 0), thickness: 1pt))
        line((x, y1 - 0.4), (x + cell_w * 2, y1 - 0.1), stroke: (paint: rgb(255, 100, 0), thickness: 1pt))
      }

      // Array dopo step 2
      let values2 = (16, 3, 12, 7, 20, 4, 14, 8)
      let y2 = -row_gap * 2
      for i in range(8) {
        let x = i * cell_w
        let fill_color = if calc.rem(i, 4) == 0 { rgb(220, 200, 250) } else { rgb(240, 240, 240) }
        rect((x, y2), (x + cell_w, y2 + cell_h), fill: fill_color, stroke: black)
        content((x + cell_w / 2, y2 + cell_h / 2), text(size: 8pt, weight: "bold")[#values2.at(i)])
      }

      // Thread ID step 3 (1 thread: 0)
      let x3 = cell_w / 2
      circle((x3, y2 - 0.6), radius: 0.18, fill: rgb(255, 150, 50), stroke: black)
      content((x3, y2 - 0.6), text(size: 7pt, fill: white, weight: "bold")[0])

      // Frecce
      line((x3, y2 - 0.4), (x3, y2 - 0.1), stroke: (paint: rgb(255, 100, 0), thickness: 1pt))
      line((x3, y2 - 0.4), (x3 + cell_w * 4, y2 - 0.1), stroke: (paint: rgb(255, 100, 0), thickness: 1pt))

      // Array finale
      let values3 = (36, 3, 12, 7, 20, 4, 14, 8)
      let y3 = -row_gap * 3
      for i in range(8) {
        let x = i * cell_w
        let fill_color = if i == 0 { rgb(220, 200, 250) } else { rgb(240, 240, 240) }
        rect((x, y3), (x + cell_w, y3 + cell_h), fill: fill_color, stroke: black)
        content((x + cell_w / 2, y3 + cell_h / 2), text(size: 8pt, weight: "bold")[#values3.at(i)])
      }
    })
  },
  caption: [
    Riduzione parallela con divergenza
  ],
)

=== Versione senza divergenza

Il problema principale della versione precedente è che introduceva una *$mr("warp divergency")$*:
- I thread sono raggruppati in warps ($32$ thread), dove ognuno di essi esegue la stessa istruzione.

- Il branch ``` if(tid % (2 * stride)) == 0: ``` causa una divergenza all'interno del warp:
  - I thread con ID pari eseguono l'operazione di somma.
  - I thread con ID dispari restano inattivi.

  Siccome il warp deve muoversi all'unisono, l'hardware è *costretto a serializzare l'esecuzione* del warp, causando un degrado delle prestazioni.

La *$mg("soluzione")$* consiste nel *sequential addressing*, ovvero nel riassegnare gli indici degli elementi dell'array a thread con ID consecutivi, in modo da evitare la divergenza (i thread inattivi saranno raggruppati alla fine):

#figure(
  {
    import cetz.draw: *

    cetz.canvas({
      let cell_w = 0.5
      let cell_h = 0.5
      let row_gap = 1.3

      // Etichetta Global memory
      content((-1.5, 0.25), text(size: 8pt)[Global memory])

      // Array iniziale in Global memory
      let values0 = (5, 1, 2, 0, 1, 1, 3, 0)
      for i in range(8) {
        let x = i * cell_w
        rect((x, 0), (x + cell_w, cell_h), fill: white, stroke: black)
        content((x + cell_w / 2, cell_h / 2), text(size: 8pt, weight: "bold")[#values0.at(i)])
      }

      // Etichetta Thread ID
      content((-1.5, -0.9), text(size: 8pt)[Thread ID])

      // Thread ID step 1 (4 thread: 0,1,2,3)
      for i in range(4) {
        let x = i * cell_w * 2 + cell_w / 2
        circle((x, -0.9), radius: 0.2, fill: rgb(255, 150, 50), stroke: black)
        content((x, -0.9), text(size: 7pt, fill: white, weight: "bold")[#i])

        // Frecce tratteggiate che connettono thread agli elementi
        let target_x = i * cell_w * 2
        line((x, -0.65), (target_x + cell_w / 2, -0.1), stroke: (paint: gray, thickness: 0.8pt, dash: "dashed"))
        line((x, -0.65), (target_x + cell_w * 1.5, -0.1), stroke: (paint: gray, thickness: 0.8pt, dash: "dashed"))
      }

      // Array dopo step 1 (stride 2)
      let values1 = (4, none, 7, none, 5, none, 5, none)
      let y1 = -row_gap - 0.5
      for i in range(8) {
        let x = i * cell_w
        let val = values1.at(i)
        let fill_color = if val != none and calc.rem(i, 2) == 0 { rgb(220, 200, 250) } else { white }
        rect((x, y1), (x + cell_w, y1 + cell_h), fill: fill_color, stroke: black)
        if val != none {
          content((x + cell_w / 2, y1 + cell_h / 2), text(size: 8pt, weight: "bold")[#val])
        }
      }

      // Thread ID step 2 (2 thread: 0,2)
      for i in (0, 2) {
        let x = i * cell_w * 2 + cell_w / 2
        circle((x, y1 - 0.5), radius: 0.2, fill: rgb(255, 150, 50), stroke: black)
        content((x, y1 - 0.5), text(size: 7pt, fill: white, weight: "bold")[#(i / 2)])

        // Frecce tratteggiate
        line((x, y1 - 0.25), (x, y1 - 0.1), stroke: (paint: gray, thickness: 0.8pt, dash: "dashed"))
        line((x, y1 - 0.25), (x + cell_w * 2, y1 - 0.1), stroke: (paint: gray, thickness: 0.8pt, dash: "dashed"))
      }

      // Array dopo step 2 (stride 4)
      let values2 = (11, none, none, none, 14, none, none, none)
      let y2 = y1 - row_gap
      for i in range(8) {
        let x = i * cell_w
        let val = values2.at(i)
        let fill_color = if val != none { rgb(220, 200, 250) } else { white }
        rect((x, y2), (x + cell_w, y2 + cell_h), fill: fill_color, stroke: black)
        if val != none {
          content((x + cell_w / 2, y2 + cell_h / 2), text(size: 8pt, weight: "bold")[#val])
        }
      }

      // Thread ID step 3 (1 thread: 0)
      let x3 = cell_w / 2
      circle((x3, y2 - 0.5), radius: 0.2, fill: rgb(255, 150, 50), stroke: black)
      content((x3, y2 - 0.5), text(size: 7pt, fill: white, weight: "bold")[0])

      // Frecce tratteggiate
      line((x3, y2 - 0.25), (x3, y2 - 0.1), stroke: (paint: gray, thickness: 0.8pt, dash: "dashed"))
      line((x3, y2 - 0.25), (x3 + cell_w * 4, y2 - 0.1), stroke: (paint: gray, thickness: 0.8pt, dash: "dashed"))

      // Array finale (stride 8)
      let values3 = (25, none, none, none, none, none, none, none)
      let y3 = y2 - row_gap
      for i in range(8) {
        let x = i * cell_w
        let val = values3.at(i)
        let fill_color = if val != none { rgb(220, 200, 250) } else { white }
        rect((x, y3), (x + cell_w, y3 + cell_h), fill: fill_color, stroke: black)
        if val != none {
          content((x + cell_w / 2, y3 + cell_h / 2), text(size: 8pt, weight: "bold")[#val])
        }
      }
    })
  },
  caption: [
    Riduzione parallela con *sequential addressing* (senza divergenza).\
    I thread con ID consecutivi lavorano su indici consecutivi, evitando warp divergence.\
  ],
)

#informalmente()[
  L'idea è che cambiare come vengono associati i thread agli indici della struttura dati.
]

```Python
@cuda.jit
def blockParReduce_no_div(in_arr, out_arr, n):
    tid = cuda.threadIdx.x
    idx = cuda.grid(1)
    # Indirizzo di partenza del blocco nelal struttura originale
    base = cuda.blockIdx.x * cuda.blockDim.x
    if (base + tid) >= n: return

    stride = 1
    while stride < cuda.blockDim.x:
        index = 2 * stride * tid
        if index < cuda.blockDim.x:
            if (base + index + stride) >= n: continue
            in_arr[base + index] += in_arr[base + index + stride]
        cuda.syncthreads()
        stride *= 2

    if tid == 0:
        out_arr[cuda.blockIdx.x] = in_arr[base]
```

== Parallel pattern: scan

L'operazione *parallel scan* (chiamata anche prefix-sum) considera un operazione binaria associativa $xor$ e un array di input $a= [x_0, x_1, ..., x_(n-1)]$. Risultato:
$
  b = [a, (a_0 xor a_1), dots, (a_0 xor a_1 xor dots xor a_(n-1))]
$

#nota()[
  Un'operatore $xor$ di scan deve avere le seguenti caratteristiche:
  - *commutativo*: $a xor b = b xor a$
  - *associativo*: $(a xor b) xor c = a xor (b xor c)$

  Tramite queste due proprietà gli elementi possono essere riordinati e combinati in qualasiasi modo senza alterare il risultato finale, ottimo per il calcolo parallelo.
]

=== Scan su big data

#informalmente()[
  In caso di un array di grandi dimensioni può essere necessario *dividere l'array in blocchi più piccoli*, eseguire la scan su ciascun blocco in parallelo, e poi combinare i risultati dei blocchi per ottenere il risultato finale.
]


#figure(
  {
    import cetz.draw: *

    cetz.canvas({
      let block_w = 1.2
      let block_h = 0.5
      let small_h = 0.35

      // PASSO 1: Array iniziale di valori arbitrari
      content((1.5, 0.8), anchor: "west", text(size: 9pt, weight: "bold")[Array iniziale])

      // Array grande diviso in 4 sezioni visibili
      for i in range(4) {
        rect((i * block_w, 0), ((i + 1) * block_w, block_h), fill: rgb(12, 160, 220), stroke: (thickness: 1.5pt))
      }

      // Frecce che vanno verso i blocchi divisi
      for i in range(4) {
        let x = i * block_w + block_w / 2
        line((x, -0.15), (x, -0.5), mark: (end: ">"), stroke: (thickness: 1pt))
      }

      // PASSO 2: Scan Block (blocchi separati)
      let y1 = -1.0
      content((0, y1 + 0.35), anchor: "west", text(size: 8pt, weight: "bold")[Scan Block 0])
      content((block_w, y1 + 0.35), anchor: "west", text(size: 8pt, weight: "bold")[Scan Block 1])
      content((2 * block_w, y1 + 0.35), anchor: "west", text(size: 8pt, weight: "bold")[Scan Block 2])
      content((4.2 * block_w, y1 + 0.25), anchor: "west", text(size: 8pt, weight: "bold")[Scan Block ])

      for i in range(4) {
        rect((i * block_w, y1), ((i + 1) * block_w, y1 + block_h), fill: rgb(120, 160, 220), stroke: (thickness: 1.5pt))
      }

      // Frecce dai blocchi all'array ausiliario
      let y2 = y1 - 1.0
      for i in range(4) {
        let x = i * block_w + block_w / 2
        line((x, y1 - 0.1), (1.8 + i * 0.4, y2 + 0.5), mark: (end: ">"), stroke: (thickness: 1pt, dash: "dashed"))
      }

      // PASSO 3: Store Block Sums in Auxiliary Array
      content((0, y2 + 0.85), anchor: "west", text(size: 8pt, weight: "bold")[])

      // Rettangolo di sfondo verde
      rect((1.4, y2 - 1.5), (3.4, y2 + 0.7), fill: rgb(200, 230, 200, 80), stroke: none)

      // Array ausiliario (piccolo, 4 celle)
      for i in range(4) {
        let fill_col = if i == 0 { white } else { rgb(120, 160, 220) }
        rect((1.8 + i * 0.4, y2), (1.8 + (i + 1) * 0.4, y2 + small_h), fill: fill_col, stroke: black)
      }

      // Freccia verso scan delle somme
      let y3 = y2 - 0.8
      line((2.4, y2 - 0.1), (2.4, y3 + 0.5), mark: (end: ">"), stroke: (thickness: 1pt))

      // PASSO 4: Scan Block Sums
      content((3.5, y3 + 0.55), anchor: "west", text(size: 8pt, weight: "bold")[Somma dei blocchi])

      for i in range(4) {
        rect((1.8 + i * 0.4, y3), (1.8 + (i + 1) * 0.4, y3 + small_h), fill: rgb(120, 160, 220), stroke: black)
      }

      // Frecce che tornano verso i blocchi finali
      let y4 = y3 - 1.0
      for i in range(4) {
        let x_src = 1.8 + i * 0.4 + 0.2
        let x_dst = i * block_w + block_w / 2
        line((x_src, y3 - 0.1), (x_dst, y4 + 0.65), mark: (end: ">"), stroke: (thickness: 1pt, dash: "dashed"))
      }

      // PASSO 5: Add Scanned Block Sums
      content((5, y4 + 0.3), anchor: "west", text(size: 8pt, weight: "bold")[Somma blocchi + offset])
      content((0, y4 + 0.3), anchor: "west", text(size: 8pt, weight: "bold")[values of Scanned Blocks 1-n])

      for i in range(4) {
        rect((i * block_w, y4), ((i + 1) * block_w, y4 + block_h), fill: rgb(120, 160, 220), stroke: (thickness: 1.5pt))
      }

      // Freccia finale verso array risultante
      for i in range(4) {
        let x = i * block_w + block_w / 2
        line((x, y4 - 0.15), (x, y4 - 0.5), mark: (end: ">"), stroke: (thickness: 1.5pt, paint: rgb(0, 120, 0)))
      }

      // PASSO 6: Final Array
      let y5 = y4 - 1.1
      content((1.5, y5 - 0.3), anchor: "west", text(size: 9pt, weight: "bold")[Array finale])

      for i in range(4) {
        rect((i * block_w, y5), ((i + 1) * block_w, y5 + block_h), fill: rgb(12, 160, 220), stroke: (thickness: 1.5pt))
      }
    })
  },
  caption: [
    Schema della parallel scan per array di grandi dimensioni.
  ],
)
Passaggi:
1. Ogni blocco calcola una *scan locale* e produce un *valore di somma parziale* (l'ultimo elemento della scan locale). Il risultato è che ogni blocco ha i numeri progressivi corretti al suo interno, ma non tiene conto delle somme dei blocchi precedenti.

  #nota()[
    é importante che i blocchi in cui viene partizionato l'array stiano in *shared memory* per massimizzare le prestazioni.
  ]

2. Le somme parziali di ogni blocco vengono raccolte in un array ausiliario più piccolo. Si esegue una *scan* su questo array. Si ottiene così un array di *offset* che rappresentano qual'è l'offset di ogni blocco.

3. Infine, viene lanciato un kernel in cui ogni thread del blocco $K$ aggiunge l'offset corrispondente al proprio elemento. Si ottiene così l'array finale corretto.


Fissando $N = "BLOCKSIZE"$, la *complessità temporale* di questa strategia è la seguente: Su ogni blocco viene eseguita una scan locale. Siccome ad ogni passo viene dimezzato il numero di thread attivi, la complessità temporale è pari a *$Theta(log(N))$*.

Tuttavia, andando a misurare la *work effeciency*, ovvero il numero totale di operazioni eseguite da tutti i thread (inclusi anche i calcoli inutili), risulta essere *$O(N log(N))$*, che è peggiore della complessità sequenziale *$O(N)$*.

=== Prefix-Sum con work efficiency

Esiste una versione *$mg("work efficinty")$* dell'algoritmo. Essa si basa sul concetto di albero binario bilanciato e si divide in due fasi distinte: *Up-Sweep* (Riduzione) e *Down-Sweep*.

*Up-Sweep*: Si costruisce un albero binario bilanciato sopra l'array di input, dove ogni nodo rappresenta la somma dei suoi figli. Si parte dai nodi foglia (gli elementi dell'array) e si risale fino alla radice (prefix-sum totale), calcolando le somme parziali.

*Down-Sweep*: Si ripercorre l'albero dalla radice verso le foglie per distribuire le somme e calcolare i prefissi.
- Il valore della radice viene inizializzato a $0$ (vincolo della scan esclusiva).

- Ad ogni passo, un nodo _padre_ possiede un certo valore $P$ e due _figli_: sinistro ($L$) e destro ($R$).

  - Il _figlio_ sinistro eredita il valore del genitore: $"New"_L = P$.

  - Il _figlio_ destro riceve la somma del genitore più il valore vecchio del figlio sinistro: $"New"_R = P + "Old"_L$.
Alla fine di questa fase, la radice contiene la somma totale dell'array.

#figure(
  {
    import cetz.draw: *

    cetz.canvas({
      let cell_w = 0.5
      let cell_h = 0.4
      let level_gap = 0.9
      let column_offset = 5.5

      // === UP-SWEEP (REDUCE PHASE) - COLONNA SINISTRA ===
      content((1, 1.0), anchor: "west", text(size: 10pt, weight: "bold")[Up-Sweep])

      // Livello 0 (input array)
      let y0 = 0
      let values0 = (3, 1, 7, 0, 4, 1, 6, 3)
      for i in range(8) {
        let x = i * cell_w
        rect((x, y0), (x + cell_w, y0 + cell_h), fill: rgb(200, 220, 255), stroke: black)
        content((x + cell_w / 2, y0 + cell_h / 2), text(size: 8pt)[#values0.at(i)])
      }

      // Livello 1 (stride 1)
      let y1 = y0 - level_gap
      let values1 = (3, 4, 7, 7, 4, 5, 6, 9)
      for i in range(8) {
        let x = i * cell_w
        let fill = if calc.rem(i, 2) == 1 { rgb(255, 200, 200) } else { rgb(200, 220, 255) }
        rect((x, y1), (x + cell_w, y1 + cell_h), fill: fill, stroke: black)
        content((x + cell_w / 2, y1 + cell_h / 2), text(size: 8pt)[#values1.at(i)])
      }

      // Frecce livello 0 -> 1
      for i in range(4) {
        let x_left = i * 2 * cell_w + cell_w / 2
        let x_right = (i * 2 + 1) * cell_w + cell_w / 2
        let x_target = x_right
        line(
          (x_left, y0 - 0.05),
          (x_target, y1 + cell_h + 0.05),
          stroke: (paint: red, thickness: 1pt),
          mark: (end: ">"),
        )
        line((x_right, y0 - 0.05), (x_target, y1 + cell_h + 0.05), stroke: (paint: red, thickness: 1pt))
      }

      // Livello 2 (stride 2)
      let y2 = y1 - level_gap
      let values2 = (3, 4, 7, 11, 4, 5, 6, 14)
      for i in range(8) {
        let x = i * cell_w
        let fill = if calc.rem(i, 4) == 3 { rgb(255, 200, 200) } else if calc.rem(i, 2) == 1 {
          rgb(220, 220, 220)
        } else { rgb(200, 220, 255) }
        rect((x, y2), (x + cell_w, y2 + cell_h), fill: fill, stroke: black)
        content((x + cell_w / 2, y2 + cell_h / 2), text(size: 8pt)[#values2.at(i)])
      }

      // Frecce livello 2 -> 1
      for i in range(2) {
        let x_left = (i * 4 + 1) * cell_w + cell_w / 2
        let x_right = (i * 4 + 3) * cell_w + cell_w / 2
        line((x_left, y1 - 0.05), (x_right, y2 + cell_h + 0.05), stroke: (paint: red, thickness: 1pt), mark: (end: ">"))
        line((x_right, y1 - 0.05), (x_right, y2 + cell_h + 0.05), stroke: (paint: red, thickness: 1pt))
      }

      // Livello 3 (stride 4) - radice
      let y3 = y2 - level_gap
      let values3 = (3, 4, 7, 11, 4, 5, 6, 25)
      for i in range(8) {
        let x = i * cell_w
        let fill = if i == 7 { rgb(255, 200, 200) } else if calc.rem(i, 4) == 3 { rgb(220, 220, 220) } else if (
          calc.rem(i, 2) == 1
        ) { rgb(220, 220, 220) } else { rgb(200, 220, 255) }
        rect((x, y3), (x + cell_w, y3 + cell_h), fill: fill, stroke: black)
        content((x + cell_w / 2, y3 + cell_h / 2), text(size: 8pt)[#values3.at(i)])
      }

      // Frecce livello 3 -> 2
      let x_left = 3 * cell_w + cell_w / 2
      let x_right = 7 * cell_w + cell_w / 2
      line((x_left, y2 - 0.05), (x_right, y3 + cell_h + 0.05), stroke: (paint: red, thickness: 1pt), mark: (end: ">"))
      line((x_right, y2 - 0.05), (x_right, y3 + cell_h + 0.05), stroke: (paint: red, thickness: 1pt))

      // === DOWN-SWEEP (DISTRIBUTE PHASE) - COLONNA DESTRA ===
      content((column_offset + 1, 1.0), anchor: "west", text(size: 10pt, weight: "bold")[Down-Sweep])

      // Livello 0 down (inizializza radice a 0)
      let dy0 = y0
      let dvalues0 = (3, 4, 7, 11, 4, 5, 6, 0)
      for i in range(8) {
        let x = column_offset + i * cell_w
        let fill = if i == 7 { rgb(200, 255, 200) } else if calc.rem(i, 4) == 3 { rgb(220, 220, 220) } else if (
          calc.rem(i, 2) == 1
        ) { rgb(220, 220, 220) } else { rgb(200, 220, 255) }
        rect((x, dy0), (x + cell_w, dy0 + cell_h), fill: fill, stroke: black)
        content((x + cell_w / 2, dy0 + cell_h / 2), text(size: 8pt)[#dvalues0.at(i)])
      }

      // Livello 1 down
      let dy1 = dy0 - level_gap
      let dvalues1 = (3, 4, 7, 0, 4, 5, 6, 11)
      for i in range(8) {
        let x = column_offset + i * cell_w
        let fill = if i == 3 or i == 7 { rgb(200, 255, 200) } else if calc.rem(i, 2) == 1 { rgb(220, 220, 220) } else {
          rgb(200, 220, 255)
        }
        rect((x, dy1), (x + cell_w, dy1 + cell_h), fill: fill, stroke: black)
        content((x + cell_w / 2, dy1 + cell_h / 2), text(size: 8pt)[#dvalues1.at(i)])
      }

      // Frecce down livello 0 -> 1 (solo una coppia: indici 3 e 7)
      let x_parent = column_offset + 7 * cell_w + cell_w / 2
      let x_left = column_offset + 3 * cell_w + cell_w / 2
      let x_right = column_offset + 7 * cell_w + cell_w / 2
      line(
        (x_parent, dy0 - 0.05),
        (x_left, dy1 + cell_h + 0.05),
        stroke: (paint: green.darken(20%), thickness: 1pt),
        mark: (end: ">"),
      )
      line(
        (x_parent, dy0 - 0.05),
        (x_right, dy1 + cell_h + 0.05),
        stroke: (paint: green.darken(20%), thickness: 1pt),
        mark: (end: ">"),
      )

      // Livello 2 down
      let dy2 = dy1 - level_gap
      let dvalues2 = (3, 0, 7, 4, 4, 0, 6, 16)
      for i in range(8) {
        let x = column_offset + i * cell_w
        let fill = if i == 1 or i == 3 or i == 5 or i == 7 { rgb(200, 255, 200) } else { rgb(200, 220, 255) }
        rect((x, dy2), (x + cell_w, dy2 + cell_h), fill: fill, stroke: black)
        content((x + cell_w / 2, dy2 + cell_h / 2), text(size: 8pt)[#dvalues2.at(i)])
      }

      // Frecce down livello 1 -> 2 (due coppie: indici (1,3) e (5,7))
      for i in (0, 1) {
        let parent_idx = if i == 0 { 3 } else { 7 }
        let left_idx = if i == 0 { 1 } else { 5 }
        let right_idx = if i == 0 { 3 } else { 7 }

        let x_parent = column_offset + parent_idx * cell_w + cell_w / 2
        let x_left = column_offset + left_idx * cell_w + cell_w / 2
        let x_right = column_offset + right_idx * cell_w + cell_w / 2
        line(
          (x_parent, dy1 - 0.05),
          (x_left, dy2 + cell_h + 0.05),
          stroke: (paint: green.darken(20%), thickness: 1pt),
          mark: (end: ">"),
        )
        line(
          (x_parent, dy1 - 0.05),
          (x_right, dy2 + cell_h + 0.05),
          stroke: (paint: green.darken(20%), thickness: 1pt),
          mark: (end: ">"),
        )
      }

      // Livello 3 down (risultato finale - exclusive scan)
      let dy3 = dy2 - level_gap
      let dvalues3 = (0, 3, 4, 11, 11, 15, 16, 22)
      for i in range(8) {
        let x = column_offset + i * cell_w
        rect((x, dy3), (x + cell_w, dy3 + cell_h), fill: rgb(200, 255, 200), stroke: black + 1.5pt)
        content((x + cell_w / 2, dy3 + cell_h / 2), text(size: 8pt, weight: "bold")[#dvalues3.at(i)])
      }

      // Frecce down livello 2 -> 3 (quattro coppie: (0,1), (2,3), (4,5), (6,7))
      for i in range(4) {
        let left_idx = i * 2
        let right_idx = i * 2 + 1
        let parent_idx = right_idx

        let x_parent = column_offset + parent_idx * cell_w + cell_w / 2
        let x_left = column_offset + left_idx * cell_w + cell_w / 2
        let x_right = column_offset + right_idx * cell_w + cell_w / 2
        line(
          (x_parent, dy2 - 0.05),
          (x_left, dy3 + cell_h + 0.05),
          stroke: (paint: green.darken(20%), thickness: 1pt),
          mark: (end: ">"),
        )
        line(
          (x_parent, dy2 - 0.05),
          (x_right, dy3 + cell_h + 0.05),
          stroke: (paint: green.darken(20%), thickness: 1pt),
          mark: (end: ">"),
        )
      }

      // Etichette
      content((4.5, y0 + cell_h / 2), anchor: "west", text(size: 7pt, fill: gray)[Input])
      content((4.5, y3 + cell_h / 2), anchor: "west", text(size: 7pt, fill: gray)[Root])
      content((column_offset + 4.5, dy0 + cell_h / 2), anchor: "west", text(size: 7pt, fill: gray)[Root = 0])
      content((column_offset + 4.5, dy3 + cell_h / 2), anchor: "west", text(size: 7pt, fill: gray)[Output])
    })
  },
  caption: [
    Algoritmo work-efficient di scan basato su albero binario.
  ],
)

Utilizzando questa versione, il numero di operazioni è stato ridotto a $2(n-1) = O(N)$. Il tempo di esecuzione rimane *$Theta(log(N))$*, in quanto l'altezza dell'albero binario è logaritmica rispetto al numero di elementi ($log_2 N$).

== Operazioni atomiche

Le *operazioni atomiche* sono operazioni di lettura/scrittura su una variabile condivisa che vengono eseguite in modo *indivisibile*. Ciò significa che una volta iniziata un'operazione atomica, nessun altro thread può interferire con essa fino a quando non è stata completata.

Questa indivisibilità viene garantita dall'hardware, il quale assicura che le operazioni concorrenti vengano serializzate.
#nota()[
  L'ordine di esecuzione tuttavia è imprevedibile.
]

Un'operazione di read-modify-write (leggere il vecchio valore, calcolarne uno nuovo e sovrascriverlo), viene tradotta in una singola istruzione hardware su un particolare indirizzo di memoria. L'hardware assicura che nessun altro thread possa eseguire un'operazione di read-modify-write sulla stessa locazione di memoria, fino a quando l'operazione precedente non è terminata. Solitamente le richieste vengono gestite tramite una coda.

Numba, mette a disposizioni le operazioni atomiche attraverso il pacchetto ``` numba.cuda.atomic ```. Alcune operazioni:

#figure(
  table(
    columns: 4,
    align: center,
    [*Operazione*], [*Firma*], [*Tipi supportati*], [*Scopo*],
    [`atomic.add` \ `atomic.sub`],
    [`add(ary, idx, val)` \ `sub(ary, idx, val)`],
    [int32, int32, float32, float64],
    [Somma e sottrazione atomica \ $"ary"["idx"] <- "ary"["idx"]+"val"$],

    [`atomic.min` \ `atomic.max`],
    [`max(ary, idx, val)` \ `min(ary, idx, val)` ],
    [int32, uint32, int64, uint64, float32, float64],
    [Massimo e minimo atomico],

    [`atomic.and_` \ `atomic.or_` \ `atomic.xor` ],
    [`and_(ary, idx, val)` \ `or_(ary, idx, val)` \ `xor(ary, idx, val)`],
    [int32, uint32, int64, uint64],
    [Scambio condizionato],

    [`atomic.exch`],
    [`exch(ary, idx, val)`],
    [int32, uint32, int64, uint64, float32],
    [Scambio non condizionato. Sostituitsce $"ary"["idx"]$ con $"val"$, restituisce il vecchio valore],

    [`atomic.cas`],
    [`cas(ary, idx, old, val)`],
    [int32, uint32, int64, uint64, float32],
    [Condizione atomica, if $"ary"["idx"] == "old"$ scrive $"val"$],
  ),
  caption: [Operazioni atomiche disponibili in Numba CUDA],
)
#esempio()[
  Un esempio di utilizzo può essere l'istogramma di un testo: contare le occorrenze di ogni lettera.
]
