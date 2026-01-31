#import "../template.typ": *

== Parallel reduction

La *reduction* è un'operazione che va a sommare gli elementi di un array di grandi diemensioni.
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
  L'array di input deve avere una dimensione pari ad una potenza di $2$ (es. $2^k$). In caso contrario, occorre *padding* l'array con zeri fino a raggiungere la dimensione successiva pari ad una potenza di $2$.
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
        content((x + cell_w/2, cell_h/2), text(size: 8pt, weight: "bold")[#values0.at(i)])
      }
      
      // Thread ID step 1 (4 thread: 0,1,2,3)
      for i in range(4) {
        let x = i * cell_w * 2 + cell_w/2
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
        content((x + cell_w/2, y1 + cell_h/2), text(size: 8pt, weight: "bold")[#values1.at(i)])
      }
      
      // Thread ID step 2 (2 thread: 0,1)
      for i in range(2) {
        let x = i * cell_w * 4 + cell_w/2
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
        content((x + cell_w/2, y2 + cell_h/2), text(size: 8pt, weight: "bold")[#values2.at(i)])
      }
      
      // Thread ID step 3 (1 thread: 0)
      let x3 = cell_w/2
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
        content((x + cell_w/2, y3 + cell_h/2), text(size: 8pt, weight: "bold")[#values3.at(i)])
      }
    })
  },
  caption: [
    Riduzione parallela con divergenza
  ]
)

=== Versione senza divergenza

Il problema principale della versione precedente è che introduceva una *$mr("warp divergency")$*: 
- I thread sono raggruppati in warps ($32$ thread), dove ognuno di essi esegue la stessa istruzione.

- Il branch ``` if(tid % (2 * stride)) == 0: ``` causa una divergenza all'interno del warp:
  - I thread con ID pari eseguono l'operazione di somma.
  - I thread con ID dispari restano inattivi.

  Siccome il warp deve muoversi all'unisono, l'hardware è *costretto a serializzare l'esecuzione* del warp, causando un degrado delle prestazioni.

La *$mg("soluzione")$* consiste nel *sequential addressing*, ovvero nel riassegnare gli indici degli elementi dell'array a thread con ID consecutivi, in modo da evitare la divergenza ( i thread inattivi saranno raggruppati alla fine):

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
        content((x + cell_w/2, cell_h/2), text(size: 8pt, weight: "bold")[#values0.at(i)])
      }
      
      // Etichetta Thread ID
      content((-1.5, -0.9), text(size: 8pt)[Thread ID])
      
      // Thread ID step 1 (4 thread: 0,1,2,3)
      for i in range(4) {
        let x = i * cell_w * 2 + cell_w/2
        circle((x, -0.9), radius: 0.2, fill: rgb(255, 150, 50), stroke: black)
        content((x, -0.9), text(size: 7pt, fill: white, weight: "bold")[#i])
        
        // Frecce tratteggiate che connettono thread agli elementi
        let target_x = i * cell_w * 2
        line((x, -0.65), (target_x + cell_w/2, -0.1), stroke: (paint: gray, thickness: 0.8pt, dash: "dashed"))
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
          content((x + cell_w/2, y1 + cell_h/2), text(size: 8pt, weight: "bold")[#val])
        }
      }
      
      // Thread ID step 2 (2 thread: 0,2)
      for i in (0, 2) {
        let x = i * cell_w * 2 + cell_w/2
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
          content((x + cell_w/2, y2 + cell_h/2), text(size: 8pt, weight: "bold")[#val])
        }
      }
      
      // Thread ID step 3 (1 thread: 0)
      let x3 = cell_w/2
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
          content((x + cell_w/2, y3 + cell_h/2), text(size: 8pt, weight: "bold")[#val])
        }
      }
    })
  },
  caption: [
    Riduzione parallela con *sequential addressing* (senza divergenza).\
    I thread con ID consecutivi lavorano su indici consecutivi, evitando warp divergence.\
  ]
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



== Prefix sum

Sequenziale: complessità linerare, in o(n) passi arriviamo alla soluzione. 

Quando ho un array di grandi diemensioni possono fare una scan sui singoli blocchi. Una volta realizzata la scan sui singolo blocchi possono andare a metterli assieme. 

posso lavolare parallelamente sui blocchi. 

Analisi dell'efficienza: 
- L'algoritmo non è work efficient. 
- la profondità è logaritmica, ma il numero di operazioni è lineare, complessit totale è $O(n log n)$.

La soluzione è un sistema work efficient. L'idea è usare un albero bianrio attraverso 2 passate otteniamo un risultato lineare. 

//Aggiungere immagine
sweep tree binary tree

== Operazioni atomiche

Operazione di lettura/scrittura in cui convergono molti thread e deve essere garantita dall'hardware che l'operaizoni sia atomica. 

Possiamo fare tante operazioni atomiche di diversa natura. Le operazioni atomiche sono: 
- add
- sub 
- in
- dec
- min 
- max

Istogramma di testo. Vogliamo raccogliere le frequenze di un item. Dobbiamo usare le operazioni atomiche per la scrittura. 

Nell'immagine ogni thread si occupa di un elemento, tutti i thread scrivono dentro la stessa struttura dati. 

