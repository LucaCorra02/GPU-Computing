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

=== Versione con divergenza

In una prima versione potremmo dividere i thread in due _gruppi_, attraverso il loro ID. In particolare, i thread con ID pari eseguono la somma del loro elemento con quello del thread successivo (ID dispari). 

Ad ogni passo lavora sola la metà dei thread, rispetto al numero di elementi ancora da sommare. 

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



i thread id pari prendono il suo elemento e quello del thread successivo ed eseguie la somma. il numero di thread che operano dimezza ad ogni passo. 

Ci focalizziamo sul blocco (porzione dell'array complessivo). Cioè lavoriamo a livello di blocco. L'idea è prendere un oggetto grande dividerlo in blocchi e successivamente riunire i risultati parziali, inoltre sono altamente sincronizzati i thread in un blocco. S
#nota()[
  Serve sincronizzazione tra uno step è l'altro. Servono che tutti i risultati dello step intermedio siano terminati, per questo lavor a livello di blocco. 
]

``` syncthread``` è dentro il for. Tutti i thread del blocco arrivano a questa barriera di iterazione, alla prossiam iterazione i dati sono aggiornati. 

#attenzione()[
  Il codice contiene un branch, quindi abbiamo una divergenza. Non è ottimale
]

il ``` blockid``` sta puntanto ad una porzione dell'array originale. in array out vengono memorizzato tutte le somme di ogni singolo blocco. 

Nella seconda immagine cambiano gli indici dei thread (meno divergenza).
#esempio()[
  Dato un array iniziale di dim $8$.
  l'idea iniziale è fare una mappa 1:1 tuttavia porta delle divergenze

  l'idea è rigiocare gli indici, non è detto che il thread lavori sull'indice corrispondente. 

  il thread id è sempre lo stesso ma capire quali stanno lavorando è importante. Lo stride è diverso da un indicizzazione sequeziale. lo stride rappresenta quale altro valore devo prendere

  in questo caso lo stride aumenta di due ad ogni iterazione: 
  - prima iterazione sommo pari e dispari threadID a 2 a due
  - seconda iterazione sommo a con offset 4
  - ultima oterazione ho tutta la somma dell'array nel thread $0$.
]

L'idea è che cambio come associo i thread agli indici della struttura dati. 

= CUDA


https://dournac.org/info/gpu_sum_reduction

//TODO guardare il PDF del labolatorio
//Aggiungere esempio reduction

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

