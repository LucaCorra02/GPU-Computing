#import "../template.typ": *

= Modello di programmazione CUDA

== Architettura GPU

Una GPU è composta da molti streaming multiprocessors (*SMs*). Ognuno di essi contiene molte unità funzionali (un certo blocco di core). Essi dispongono di una memoria privata, cache e register file. \
A loro volta gli SMs, sono raggruppati in cluster, chiamati Graphics processing clusters (*GPCs*). Essi lavorano condividendo un'unica memoria (L2, cache). La GPU non è altro che un insieme di GPCs.

La CPU (host) è connessa alla GPU tramite degli appositi canali. 

== Thread in CUDA

Pensare in parallelo, significa avere chiaro quali *feature* la *GPU* espone al programmatore: 
- è essenziale conoscere l'architettura della GPU per scalare su migliaia di thread come fosse uno. 
- gestire la cache, in modo da sfruttare il *principio di località*.
- conoscere lo *scheduling* dei blocchi di thread. Se il blocco di thread è molto esoso in termini di risorse, potrebbe essere eseguito in modo singolare durante un certo istante di tempo.
- gestire le *sincronizzazioni*. I thread a volte potrebbero dover cooperare nella GPU. Bisogna effettuare una sincronizzazione all'interno dei blocchi logici di thread. 

CUDA, permette al programmatore di gestire i thread e la memoria dati.
#attenzione[
  Le operazioni di lancio del kernel, sono sempre asincrone. Mentre le operazioni in memoria per definizione sono sincrone, in modo da garantire l'integrità dei dati.   
]

Infine il compilatore (_nvcc_) deve generare codice eseguibile per host(linguaggio _C_ o altro) e device (_Cuda C_), l'output di questa fase prende il nome di *fat binary*.   

== Processing Flow

In generale, lo schema da seguire è sempre lo stesso: 
- copiare i dati da elaborare dalla CPU alla GPU
- caricare ed eseguire il programma in GPU. Esso caricare i dati nella cache della GPU, in modo da migliorare le performance. 
- copiare i dati dalla memoria della GPU alla memoria della CPU.  

== Gerarchia dei thread

CUDA presenta una *gerarchia astratta* di thread, strutturata su due livelli: 
- *grid*: una griglia ordinata di blocchi
- *block*: una collezione ordinata di thread. 

La struttura a _blocchi_ permette alla GPU di distribuire il lavoro. I blocchi vengono assegnati ai vari SM disponibili. Una GPU potente con molti SM, eseguirà più blocchi contemporaneamente. Questo permette di scrivere il codice una volta sola e farlo "scalare" su hardware diverso.

#informalmente()[
  Sebbene la memoria fisica della GPU sia sempre lineare (una lunga sequenza di byte 1D), per i programmatori è difficile ragionare solo in termini lineari se il problema è geometrico. Per questo motivo si usa questa organizzazione logica. 
]

Sia le griglie che i blocchi possono avere una dimensione ($1D$, $2D$ o $3D$).
#nota[
  In generale si usa la stessa dimensione sia per le griglie che i blocchi
]
La scelta delle dimensioni avviene in base ai dati che si vuole elaborare.

Le dimensioni vengono gestite nel seguente modo:
- *$"grid"(mb(x),mr(y),mg(z))$*: 
  - $mb(x)$ = Numero di blocchi in una riga 
  - $mr(z)$ = Numero di righe di blocchi 
  - $mg(z)$ = Profondità 

- *$"block"(mb(x),mr(y),mg(z))$*: 
  - $mb(x)$ = Numero di thread in una riga del blocco  
  - $mr(z)$ = Numero di righe del blocco
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
- *$"blockIdx"(x,y,z)$* (indice del blocco all'interno della grid). Tipo ``` uint3```
- *$"threadIdx"(x,y,z)$* (indice di thread nel blocco). Tipo ``` uint3```

Tali variabili vengono pre-inizializzate e possono essere accedute all'interno del kernel. Quando un kernel viene eseguito ``` blockIdx``` e ``` threadIdx``` vengono assegnate a ogni thread da CUDA *runtime*.

=== Dati lineari (1D)

Si usa una griglia 1D, quando il dato da elaborare è un array (dato in sequenza). Per identificare un dato della struttura originale basta una cordinata $x$. 

Per ottenere un thread ID univoco globale, indipendente dalla disposizione adottata, si usa la seguente *indicizzazione*: 
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
