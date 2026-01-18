#import "../template.typ": *

= Modello di programmazione CUDA

== Architettura GPU

Una GPU è composta da molti streaming multiprocessors (*SMs*). Ognuno di essi contiene molte unità funzionali (un certo blocco di core). Essi dispongono di una memoria privata, cache e register file. \
A loro volta gli SMs, sono raggruppati in cluster, chiamati Graphics processing clusters (*GPCs*). Essi lavorano condividendo un'unica memoria (L2 cache). La GPU non è altro che un insieme di GPCs.

La CPU (host) è connessa alla GPU tramite degli appositi canali. 

== Thread in CUDA

Pensare in parallelo, significa avere chiaro quali *feature* la *GPU* espone al programmatore: 
- è essenziale conoscere l'architettura della GPU per scalare su migliaia di thread come fosse uno. 
- gestire la cache, in modo da sfruttare il *principio di località*.
- conoscere lo *scheduling* dei blocchi di thread. Se il blocco di thread è molto esoso in termini di risorse, potrebbe essere eseguito in modo singolare sulla GPU durante un certo istante di tempo.
- gestire le *sincronizzazioni*. I thread a volte potrebbero dover cooperare nella GPU. Bisogna effettuare una sincronizzazione all'interno dei blocchi logici di thread. 

CUDA permette al programmatore di gestire i thread e la memoria dati.
#attenzione[
  Le operazioni di lancio del kernel sono sempre asincrone. Mentre le operazioni in memoria, per definizione, sono sincrone. Questo permette di garantire l'integrità dei dati.   
]

Infine il compilatore (_nvcc_) deve generare codice eseguibile per host(linguaggio _C_ o altro) e device (_Cuda C_), l'output di questa fase prende il nome di *fat binary*.   

== Processing Flow

In generale, lo schema da seguire è sempre lo stesso: 
- copiare i dati da elaborare dalla CPU alla GPU
- caricare ed eseguire il programma in GPU. Caricare i dati nella cache della GPU, in modo da migliorare le performance. 
- copiare i dati dalla memoria della GPU alla memoria della CPU.  

== Gerarchia dei thread

CUDA presenta una *gerarchia astratta* di thread, strutturata su due livelli: 
- *grid*: una griglia ordinata di blocchi
- *block*: una collezione ordinata di thread. 

La struttura a _blocchi_ permette alla GPU di distribuire il lavoro. I blocchi vengono assegnati ai vari SM disponibili. Una GPU potente con molti SM eseguirà più blocchi contemporaneamente. Questo permette di scrivere il codice una volta sola e farlo "scalare" su hardware diverso.

#informalmente()[
  Sebbene la memoria fisica della GPU sia sempre lineare (una lunga sequenza di byte 1D), per i programmatori è difficile ragionare solo in termini lineari se il problema da risolvere è geometrico. Per questo motivo si usa una organizzazione logica astratta. 
]

Sia le griglie che i blocchi possono avere una *dimensione* ($1D$, $2D$ o $3D$).
#nota[
  In generale si usa la stessa dimensione sia per le griglie che i blocchi
]
La scelta del numero di dimensioni avviene in base ai dati che si vuole elaborare.

Le dimensioni vengono gestite nel seguente modo:
- *$"grid"(mb(x),mr(y),mg(z))$*: 
  - $mb(x)$ = Numero di blocchi in una riga 
  - $mr(y)$ = Numero di righe di blocchi 
  - $mg(z)$ = Profondità 

- *$"block"(mb(x),mr(y),mg(z))$*: 
  - $mb(x)$ = Numero di thread in una riga del blocco  
  - $mr(y)$ = Numero di righe del blocco
  - $mg(z)$ = Profondità 

Per ottenere il numero totale di thread basta moltiplicare tutte le dimensioni di grid e block tra di loro. 

#attenzione[
  Il numero di thread *massimo* in un blocco è *$1024$* 
]

=== Mapping

Un *blocco* è quindi un gruppo di thread che possono cooperare tra loro (anche thread in blocchi diversi) mediante due tecniche: 
- *Block-local synchronization*
- *Block-local shared memory*

#nota()[
  Tutti i thread in una griglia condividono lo stesso spazio di memoria
]

Ogni *thread è identificato univocamente* da due coordinate (sono delle vartiabili built-in):
- *$"blockIdx"(x,y,z)$* indice del blocco all'interno della grid. Tipo ``` uint3```
- *$"threadIdx"(x,y,z)$* indice di thread nel blocco. Tipo ``` uint3```

Tali variabili vengono pre-inizializzate e possono essere accedute all'interno del kernel. Quando un kernel viene eseguito ``` blockIdx``` e ``` threadIdx``` vengono assegnate a ogni thread da CUDA *runtime*.

=== Dati lineari (1D)

Si usa una griglia 1D quando il dato da elaborare è un array (dato in sequenza). Per identificare un dato della struttura originale basta una singola coordinata $x$. 

Per ottenere un thread ID univoco a livello globale, indipendente dalla disposizione logica adottata, si usa la seguente *indicizzazione*: 
$
  "ID"_{"th"} = underbrace(mr("blockIdx".x * "blockDim".x), "salta i blocchi precedenti") + underbrace(mb("threadIdx".x),"posizione locale del thread")
$

#figure(
  {
    import cetz.draw: *
    
    cetz.canvas({
      // Grid 1D container
      rect((0, 0), (14, 4), stroke: (paint: purple, thickness: 2pt), name: "grid")
      content((1.5, 3.7), text(fill: blue, weight: "bold", size: 14pt)[GRID 1D])
      
      // Block 0
      rect((0.5, 0.5), (4.2, 3.2), stroke: (paint: orange, thickness: 1.5pt), name: "block0")
      content((1.5, 3), text(fill: blue, weight: "bold")[BLOCK 1D])
      content((2.35, 2.3), text(fill: purple, weight: "bold")[BK(0)])
      
      // Threads in Block 0
      let block0_x = 0.7
      for i in range(4) {
        rect((block0_x + i * 0.85, 0.8), (block0_x + (i + 1) * 0.85 - 0.05, 1.8), 
             fill: rgb(255, 200, 200), stroke: rgb(150, 50, 50))
        content((block0_x + i * 0.85 + 0.4, 1.3), text(fill: purple, size: 8pt, weight: "bold")[TH(#i)])
      }
      
      // Block 1
      rect((4.8, 0.5), (8.5, 3.2), stroke: (paint: orange, thickness: 1.5pt), name: "block1")
      content((5.8, 3), text(fill: blue, weight: "bold")[BLOCK 1D])
      content((6.65, 2.2), text(fill: purple, weight: "bold")[BK(1)])
      
      // Threads in Block 1
      let block1_x = 5.0
      for i in range(4) {
        rect((block1_x + i * 0.85, 0.8), (block1_x + (i + 1) * 0.85 - 0.05, 1.8), 
             fill: rgb(255, 200, 200), stroke: rgb(150, 50, 50))
        content((block1_x + i * 0.85 + 0.4, 1.3), text(fill: purple, size: 8pt, weight: "bold")[TH(#i)])
      }
      
      // Block 2
      rect((9.1, 0.5), (12.8, 3.2), stroke: (paint: orange, thickness: 1.5pt), name: "block2")
      content((10.1, 3), text(fill: blue, weight: "bold")[BLOCK 1D])
      content((10.95, 2.2), text(fill: purple, weight: "bold")[BK(2)])
      
      // Threads in Block 2
      let block2_x = 9.3
      for i in range(4) {
        rect((block2_x + i * 0.85, 0.8), (block2_x + (i + 1) * 0.85 - 0.05, 1.8), 
             fill: rgb(255, 200, 200), stroke: rgb(150, 50, 50))
        content((block2_x + i * 0.85 + 0.4, 1.3), text(fill: purple, size: 8pt, weight: "bold")[TH(#i)])
      }
    
    })
  },
  caption: [
    Esempio di Grid 1D con Blocchi 1D.\
    $"grid"(3,1,1)$ e $"block"(4,1,1)$\
    Totale thread = $3*4 = 12$
  ]
)

#esempio()[
  Indicizzazione: 
  $
    "ID"_("th") = underbrace({0,1,2},"indice blocco") * 4 + underbrace({0,1,2,3}, "thread id locale")
  $
]

=== Dati piani (2D)

Adatto per mappare matrici/immagini. Servono 2 coordinate $(x,y)$. In questo caso l'indicizzazione avviene in due fasi:
- Trovare le *coordinate globali* $(x,y)$:
  Dobbiamo prima capire dove ci troviamo nella griglia globale immaginaria:
  $
    "ix" = "blockIdx".x * "blockDim".x + "threadIdx".x\
    "iy" = "blockIdx".y * "blockDim".y + "threadIdx".y
  $
- *Linearizzazione* (da 2D a 1D): Le matrici per convenzione vengono memorizzate riga per colonna (*row-major oder*), per arrivare a una determinata riga, dobbiamo "saltare" tutte le righe complete precedenti. Una volta trovate le coordinate $("ix","iy")$ dobbiamo linearizzarle. 
$
  "idx" = underbrace(mr("iy" * "larghezza-matrice"),"salta le righe precedenti") + underbrace(mb("ix"),"indice riga corrente")
$
Spesso è necessario un controllo quando la dim di griglia non collima con quella della matrice
```
  if (ix < "blockDim".x & ix < "blockDim".y) // va bene
```

#attenzione()[
  Il controllo è *obbligatorio*. Siccome il kernel accetta due dimensioni (numero di blocchi per griglia, numero di thread per blocco) può essere che la divisione logica non sia intera (approssimazione ``` ceil```). Il controllo evita accessi ``` out_of_bound``` sulla struttura originale.
]

#figure(
  {
    import cetz.draw: *
    
    cetz.canvas({
      // Formula at top
      content((3, 4.8), text(fill: black, size: 9pt)[ix = threadIdx.x + blockIdx.x × blockDim.x])
      
      // Draw grid of blocks
      let block_size = 1.3
      let gap = 0.1
      let start_x = 0.8
      let start_y = 0
      
      // Draw 4x3 grid of blocks
      for row in range(3) {
        for col in range(4) {
          let x = start_x + col * (block_size + gap)
          let y = start_y + (2 - row) * (block_size + gap)
          
          rect(
            (x, y), 
            (x + block_size, y + block_size),
            fill: rgb(180, 220, 150),
            stroke: (paint: rgb(80, 100, 60), thickness: 1.5pt)
          )
        }
      }
      
      // Highlight the target block (row 1, col 2)
      let target_x = start_x + 2 * (block_size + gap)
      let target_y = start_y + 1 * (block_size + gap)
      
      // Draw vertical dashed line
      line(
        (target_x + block_size/2, start_y - 0.3),
        (target_x + block_size/2, start_y + 3 * (block_size + gap)),
        stroke: (paint: rgb(200, 180, 0), thickness: 1.2pt, dash: "dashed")
      )
      
      // Draw horizontal dashed line
      line(
        (start_x - 0.3, target_y + block_size/2),
        (start_x + 4 * (block_size + gap), target_y + block_size/2),
        stroke: (paint: rgb(200, 180, 0), thickness: 1.2pt, dash: "dashed")
      )
      
      // Draw red dot at intersection
      circle(
        (target_x + block_size/2, target_y + block_size/2),
        radius: 0.08,
        fill: red,
        stroke: none
      )
      
      // Label (ix, iy)
      content(
        (target_x + block_size/2 + 0.7, target_y+-0.2 + block_size/2),
        text(fill: red, weight: "bold", size: 8pt)[(ix, iy)]
      )
      
      // Label nx at top right
      content(
        (start_x + 4 * (block_size + gap) + 0.4, start_y + 2.2 * (block_size + gap)),
        text(fill: black, size: 9pt)[nx]
      )
      
      // Label ny at left
      content(
        (start_x - 0.6, start_y + 1.5 * (block_size + gap)),
        text(fill: black, size: 9pt)[ny],
        angle: 90deg
      )
      
      // Label iy formula on left side
      content(
        (-0.3, start_y + 1.5 * (block_size + gap)),
        text(fill: black, size: 8pt)[iy = threadIdx.y + blockIdx.y × blockDim.y],
        angle: 90deg
      )
    })
  },
  caption: [
    Mappatura di un thread nelle coordinate globali $(mb("ix"), mr("iy"))$ di una matrice.\
    Le linee tratteggiate mostrano come vengono calcolate le coordinate globali del thread
  ]
)

#esempio(
  figure(
    {
      import cetz.draw: *
      
      cetz.canvas({
        // Axis labels
        content((-1.0, 4), text(fill: purple, weight: "bold")[Dim Y])
        content((3, 6.5), text(fill: purple, weight: "bold")[Dim X])
        
        // Arrow for dimension X
        line((0.5, 6.2), (5.5, 6.2), mark: (end: "stealth"), stroke: black)
        
        // Arrow for dimension Y
        line((-0.2, 6), (-0.2, 1), mark: (end: "stealth"), stroke: black)
        
        // Define colors for each block
        let color_b00 = rgb(255, 200, 150)  // Block (0,0) - orange
        let color_b10 = rgb(200, 255, 200)  // Block (1,0) - green
        let color_b01 = rgb(255, 150, 150)  // Block (0,1) - red
        let color_b11 = rgb(200, 200, 255)  // Block (1,1) - blue
        
        // Helper function to draw a 3x3 block with colored labels
        let draw-matrix-block(x, y, bx, by, color, label_color) = {
          let cell_size = 0.8
          
          // Draw block border
          rect((x, y - 2.4), (x + 2.4, y), stroke: (paint: label_color, thickness: 1.5pt))
          
          for ty in range(3) {
            for tx in range(3) {
              let cx = x + tx * cell_size
              let cy = y - ty * cell_size
              rect(
                (cx, cy - cell_size), 
                (cx + cell_size, cy),
                fill: color,
                stroke: black
              )
              content(
                (cx + cell_size/2, cy - cell_size/2),
                text(size: 8pt, weight: "bold")[#tx,#ty]
              )
            }
          }
          
            content((x + 1.2, y + 0.6), text(fill: label_color, size: 8pt)[blockIdx.x])
          content((x + 1.2, y + 0.3), text(fill: label_color, size: 9pt, weight: "bold")[#bx])
          
        
          
          // Block labels - Y on right side (more padding)
          content((x + 3.2, y - 1.2), text(fill: label_color, size: 8pt)[blockIdx.y])
          content((x + 3.2, y - 1.5), text(fill: label_color, size: 9pt, weight: "bold")[1])
          
          // Block labels - Y on right side  
        
        }
        
        // Draw four blocks with correct coordinates and matching label colors
        draw-matrix-block(0.5, 5.5, 0, 0, color_b00, rgb(200, 100, 50))   // Top-left - orange labels
        draw-matrix-block(3.2, 5.5, 1, 0, color_b10, rgb(50, 150, 50))    // Top-right - green labels
        draw-matrix-block(0.5, 2.3, 0, 1, color_b01, rgb(200, 50, 50))    // Bottom-left - red labels
        draw-matrix-block(3.2, 2.3, 1, 1, color_b11, rgb(50, 50, 200))    // Bottom-right - blue labels
        
        
        
        // Right side - Linearized memory representation
        content((10.7, 6.3), text(fill: blue, weight: "bold", size: 10pt)[Rappresentazione in memoria (row-major)])
        
        let draw-linear-row(x, y, start_idx, label) = {
          let cell_w = 0.45
          
          // Draw cells with appropriate colors based on block origin
          for i in range(6) {
            let cx = x + i * cell_w
            let color = if i < 3 { color_b00 } else { color_b10 }
            
            // Adjust color based on Y coordinate
            if start_idx >= 18 {
              color = if i < 3 { color_b01 } else { color_b11 }
            }
            
            rect((cx, y), (cx + cell_w, y + 0.5), fill: color, stroke: black)
            content((cx + cell_w/2, y + 0.25), text(size: 6pt, weight: "bold")[#(start_idx + i)])
          }
          
          // Label on the right
          content((x + 6 * cell_w + 0.8, y + 0.25), text(size: 8pt)[#label])
        }
        
        let y_pos = 5.5
        let x_start = 7.5
        
        // Y=0 to Y=2: Blocks (0,0) and (1,0)
        draw-linear-row(x_start, y_pos, 0, "← Y=0, X=0..5")
        
        y_pos -= 0.7
        draw-linear-row(x_start, y_pos, 6, "← Y=1, X=0..5")
        
        y_pos -= 0.7
        draw-linear-row(x_start, y_pos, 12, "← Y=2, X=0..5")
        
        // Y=3 to Y=5: Blocks (0,1) and (1,1)
        y_pos -= 0.7
        draw-linear-row(x_start, y_pos, 18, "← Y=3, X=0..5")
        
        y_pos -= 0.7
        draw-linear-row(x_start, y_pos, 24, "← Y=4, X=0..5")
        
        y_pos -= 0.7
        draw-linear-row(x_start, y_pos, 30, "← Y=5, X=0..5")
      
        })
    },
    caption: [
      Linearizzazione di una matrice con Grid 2D e Block 2D.\
      A sinistra: vista logica 2D della griglia con 4 blocchi.\
      $"Grid" 2 * 2$, $"Block" 3 * 3$ 
    ]
  )
)

/*
#figure(
  {
    import cetz.draw: *
    
    cetz.canvas({
      // Grid 2D container
      rect((0, 0), (14, 10), stroke: (paint: purple, thickness: 2pt))
      content((1.2, 9.5), text(fill: blue, weight: "bold", size: 14pt)[GRID 2D])
      
      // Helper function to draw a block with threads
      let draw-block(x, y, bx, by, label) = {
        // Block container
        rect((x, y), (x + 4, y + 4), stroke: (paint: orange, thickness: 1.5pt))
        content((x + 1.8, y + 4.2), text(fill: blue, weight: "bold")[BLOCK 2D])
        content((x + 2, y + 3.2), text(fill: purple, weight: "bold")[BK(#bx,#by)])
        
        // Draw 3x3 grid of threads
        let cell_size = 1.2
        let start_x = x + 0.3
        let start_y = y + 0.3
        
        for ty in range(3) {
          for tx in range(3) {
            let cx = start_x + tx * cell_size
            let cy = start_y + (2 - ty) * cell_size
            rect(
              (cx, cy), 
              (cx + cell_size - 0.05, cy + cell_size - 0.05),
              fill: rgb(255, 200, 200),
              stroke: rgb(150, 50, 50)
            )
            content(
              (cx + cell_size/2, cy + cell_size/2),
              text(fill: purple, size: 8pt, weight: "bold")[TH(#tx,#ty)]
            )
          }
        }
      }
      
      // First row of blocks (y = 0)
      draw-block(0.5, 4.8, 0, 0, "BK(0,0)")
      draw-block(5, 4.8, 1, 0, "BK(1,0)")
      draw-block(9.5, 4.8, 2, 0, "BK(2,0)")
      
      // Second row of blocks (y = 1) - maggiore spaziatura
      draw-block(0.5, 0.3, 0, 1, "BK(0,1)")
      draw-block(5, 0.3, 1, 1, "BK(1,1)")
      draw-block(9.5, 0.3, 2, 1, "BK(2,1)")
    })
  },
  caption: [
    Esempio di Grid 2D con Blocchi 2D.\
    $"grid"(3,2,1)$ e $"block"(3,3,1)$\
    Totale thread = $3*2*3*3 = 54 $
  ]
)
*/

== Misto (Grid 2D e blocco 1D)

Possiamo usare la seguente formula: 
$
  "ID" = underbrace(("blockIdx".x * "DimBlocco"),"salto i blocchi precedenti") + underbrace(("threadIdx".y * "LarghezzaRiga"),"salto righe nel blocco") + "threadIdx".x 
$

== Kernell di CUDA

Un kernel in CUDA ha la seguente sintassi: 
```
  kernel_name[grid,block](argument)
```
Dove: 
- *grid* = numero di blocchi che racchiudono i thread
- *block* = numero di thread nel blocco 

Prorietà di un kernel. Sono dei qualificatori: 
- $mb("__global__")$:
  - Funzione eseguita dal device
  - Può essere solamente lanciata dal kernel
  - Restituisce un tipo void 
- $mb("__device__")$: 
  - Eseguita dal device
  - Può essere chiamata solo dal device
- $mb("__host__")$: 
  - Può essere omesso
  - Computata e chiamata solo dalla CPU

  == CUDA con Numba

  Per definire un kernel in Numba, dobbiamo: 
  - Usare il decoratore ``` @cuda.jit```
  - *Non* ha valore di ritorno
  - Tutti gli output del kernel, devono essere scritti negli array passati come argomenti
  - La configurazione è specificata come $"grid" & "block"$

  ```py
  from numba import cuda
  
  @cuda.jit
  def my_kernel(a, b, out):
    ...
  
  threadsperblock = 32
  blockspergrid = (array.size + (threadsperblock - 1)) // threadsperblock
  my_kernel[blockspergrid, threadsperblock](a, b, out)
  cuda.synchronize()
  ```
  il numero totale di thread è dato da $"blockspergrid" * "threadsperblock"$. 

  #attenzione()[
    L'esecuzione del kernel avviene in modo asincrono. Lato host bisogna usare ``` cuda.synchronize()``` per aspettare i risultati dai thread. 
  ]

  Per ogni thread è possibile avere le seguenti dimensioni: 
  - ``` cuda.threadIdx``` = index del thread dentro il blocco
  - ``` cuda.blockIdx``` = indice del blocco nella griglia
  - ``` cuda.blockDim``` = dimensione del blocco 
  - ``` cuda.gridDim``` = dimensione della griglia

  Numba fornisce anche due funzioni per calcolare la *posizione assoluta* di un *thread*:
  - ``` cuda.grid(ndim)``` = *indice assoluto* del thread nella griglia
  -  ``` cuda.gridsize(ndim)``` = numero totale di thread nella griglia

=== Trasferimento dati in numba

La GPU e la CPU hanno due memorie separate. I trasferimenti di dati devono sempre seguire l'ordine $"CPU" -> "GPU" -> "CPU"$.

#nota()[
  Il trasferimento di dati può essere in molti casi il bottleneck
]

Numba trasferisce automaticamente gli array Numpy alla GPU, tuttavia lo fa in una maniera *conservativa*. Quando il kernel ha finito, i dati vengono copiati nell'host. \
Questo comportamento a volte può essere un $mr("problema")$:
- Trasferimenti non necessari -> sprecano banda

Tuttavia Numba, mette a disposizione delle API in grado di :
- Allocare la memoria sulla GPU
- Copiare i dati solamente quando serve
- Mantenere dati sul device tra kernel

=== Device Arrays

Permettono di allocare un array vuoto direttamente sull GPU 
```py
numba.cuda.device_array(
  shape,
  dtype=np.float64,
  strides=None,
  order='C',
  stream=0
)
```
il vantaggio è che la memoria viene allocata solo lato device, inoltre non vengono ricopiati automaticamente sul host.  
#nota()[
  Stesso comportamento di ``` numpy.empty()``` lato host
]

```py 
  n = 1024
  # d_out vive esclusivamente sulla GPU
  d_out = cuda.device_array((n,),dtype=np.float32)
```
*Quando usarli*: 
- Buffer di output
- Risultati intermedi
- Dati che devono rimanere sulla GPU per più kernel

=== Funzioni di trasferimento

la funzione ```py cuda.device_array_like(h_a)``` permette di *creare un device array* utilizando la dimensione di un'altro array già esistente. Solitamente si usa il seguente pattern: 
```py
# Host array
h_a = np.random.rand(1024).astype(np.float32)
# Device array, solamente output
d_b = cuda.device_array_like(h_a)
```

Per il trasferimento da *Host $->$ Device*, viene utilizzata la funzione ```py numba.cuda.to_device(obj,stream=0, copy=True, to=None)```:
- ``` to=existing_device_array```, usa memoria già allocata
- ``` stream``` = abilità il trasferimento asincrono 

Per il trasferimento Device $->$ Host, viene utilizzata la funzione ```py copy_to_host()```.

=== Esempio: Fibonacci
#link("https://colab.research.google.com/drive/1H_67B-cdnNXElwzpk9BCSY4lmvKE8eta?authuser=1#scrollTo=2XXzzmph7dQ1")[
  Lezione $2$ su colab
].
Data una matrice $v$ di dimensione $n * n$, restituire una matrice in output dove: 
$
  v[i][j] = 0 "se" "is_Fib"(i + j) = "False" \ 
  v[i][j] = 1 "se" "is_Fib"(i + j) = "True"
$

#nota()[
  - Il bound degli indici nel kernel è *obbligatorio*. A casua della funzione di ``` ceil```, potremmo eseguire più blocchi del necessario. 
  - Per quanto riguarda la divisione in blocchi, le GPU moderne eseguono i thread in gruppi indivisibili chiamati *Wrap*. Le GPU moderne preferiscono blocchi con più wrap per blocco. Solitamente si opra per blocchi da $16*16 = 256$ thread, oppure $32*8 = 256$ thread (sfrutta la coalescing della memoria).
]

```py

import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt

@cuda.jit(device=True, inline=True)
def is_perfect_square(m):
    if m < 0: return False
    tmp = int(m**0.5)
    return True if tmp * tmp == m else False

@cuda.jit(device=True, inline=True)
def is_fibonacci(v):
    if v < 0: return 0
    if is_perfect_square(5*(v**2) + 4) or is_perfect_square(5*(v**2) - 4):
        return 1
    return 0

@cuda.jit
def fibonacci_kernel_2D(out):
    #coordinate assolute del thread nella griglia
    x, y = cuda.grid(2)
    if x < out.shape[0] and y < out.shape[1]:
        val_to_check = abs(x - y)
        out[x, y] = 1 if is_fibonacci(val_to_check) else 0

def main():
    n = 1024  # Dimensione matrice (pixel)
    out = np.zeros((n, n), dtype=np.int32)
    d_out = cuda.to_device(out) 

    threads_per_block = (32, 8) # 64 thread
    #Numero di blocchi che servono a coprire la matrice
    blocks_per_grid_x = math.ceil(n / threads_per_block[0])
    blocks_per_grid_y = math.ceil(n / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    fibonacci_kernel_2D[blocks_per_grid, threads_per_block](d_out)
    cuda.synchronize()
    ris = d_out.copy_to_host()

if __name__ == "__main__":
    main()

```