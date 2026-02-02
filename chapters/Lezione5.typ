#import "../template.typ": *

== Cache su GPU

Su una GPU esistono in genere $4$ tipi di cache:
- L1: Una per ogni SM
- L2: Condivisa tra tutti gli SM della GPU
- Read only constant
- Read only texture

Le ultime due cache sono presenti su ogni SM e vengono utilizzate per migliorare le prestazioni in lettura dai rispettivi in lettura dai rispettivi spazi di memoria sul device.

Solitamente le cache L1 e L2 vengono usate per memorizzare dati in *memoria locale* e *globale*, incluso lo spilling dei registri (eccessi nell'uso di local memory).\
Inoltre, tramite la primitiva ` cudaFuncSetAttribute` è possibile a run time o compile time dividere in modo arbitrario la L1 cache, tra L1 e shared memory.

== Shared memory

A differenza della global memory (DRAM) situata al di fuori dei chip, la *shared memory* si tratta di una memoria che risiede sul chip, offrendo una bandwidth molto più alta e una minore latenza rispetto alla DRAM globale, giustificando il costo dell'overhead aggiunto.

La shared memory ha una *visibilità a livello di blocco*. I thread all'interno dello stesso blocco comunicano attraverso shared memory, per questo motivo, la shared memory ha una vita correlata con la durata di esecuzione del blocco.

#nota()[
  Ha senso usare la shared memory quanto il *tasso di accesso al singolo dato è molto elevato*, ad esempio quando i thread di un certo blocco cooperano su un blocco di dati in shared memory.
]

Siccome il *trasferimento di dati* dalla global memory alla shared memory è *distribuito* tra i thread del blocco (ogni thread trasporta un certo set di dati), è *necessaria sincronizzazione*. In quanto per iniziare la computazione è necessario che tutti i dati siano stati trasferiti, abbiamo così dei dati afffidabili (no race condition).

Maggiore è la shared memory richiesta da un kernel, minore è il numero di blocchi attivi concorrenti.

#nota()[
  Ogni SM ha una quantità limitata di shared memory che viene ripartita tra i blocchi di thread. Il numero massimo di blocchi ospitati all'interno di un SM può scendere se all'interno di un blocco vengono richieste molte risorse (registri e shared memory), in quanto esse sono condivise nell'SM.\
  Il numero massimo di blocchi non è mai fisso ma dipende dall'*occupancy* (tasso di warp attivi simltuanamente nell'SM).
]

=== Parallel access via banking

La memoria shared viene divisa in $32$ moduli della stessa ampiezza chiamati *bank*. La dimensione dei banchi dipende dalla dimensione dei dati, solitamente $4$ byte(int), ovvero una word. I banchi solitamente implementato sequenze di word.

#nota()[
  La shared memory è tagliata sulla dimensione del warp, in modo tale da aumentare l'efficienza.
]

L'accesso avviene per warp. Ogni richiesta di accesso fatta di $N$ indirizzi che riguardino $N$ distinti bank sono serviti simultaneamente.

Possiamo avere due pattern di accesso alla shared memory:
- *Acesso parallelo*: se abbiamo un warp dove ogni thread richiede al più un accesso per bank, possiamo effettuare in una sola transizione il trasferimento dati.

- *Accesso sequenziale*: si possono creare dei conflitti se diversi thread cercano di accedere allo stesso blocco. In questo caso l'hardware effetua tante transizioni quante ne sono necessarie per eliminare i conflitti

#figure(
  stack(
    dir: ttb,
    spacing: 2em,
    [
      // Accesso parallelo - nessun conflitto
      #align(center)[
        #text(size: 10pt, weight: "bold", fill: rgb(100, 180, 50))[Accesso Parallelo (senza conflitti)]
      ]
      #import cetz.draw: *
      #cetz.canvas({
        let thread_w = 0.35
        let bank_w = 0.35
        let thread_h = 0.4
        let bank_h = 0.4
        
        // Thread row
        for i in range(32) {
          let x = i * thread_w
          rect((x, 2.5), (x + thread_w, 2.5 + thread_h),
               fill: rgb(200, 220, 200),
               stroke: black)
          content((x + thread_w/2, 2.7),
                  text(size: 5pt, weight: "bold")[T#i])
        }
        
        // Arrows - ogni thread accede a un bank diverso
        for i in range(32) {
          let x = i * thread_w + thread_w/2
          line((x, 2.5), (x, 1.5),
               mark: (end: "stealth"),
               stroke: (paint: rgb(100, 180, 50), thickness: 0.8pt))
        }
        
        // Bank row
        for i in range(32) {
          let x = i * bank_w
          rect((x, 1), (x + bank_w, 1 + bank_h),
               fill: rgb(150, 200, 255),
               stroke: black)
          content((x + bank_w/2, 1.2),
                  text(size: 5pt, weight: "bold")[B#i])
        }
        
        // Label
        content((5.6, 0.5), text(size: 8pt, fill: rgb(100, 180, 50))[
          *1 transazione* - Tutti i thread accedono simultaneamente
        ])
      })
    ],
    [
      // Accesso sequenziale - conflitti
      #align(center)[
        #text(size: 10pt, weight: "bold", fill: rgb(200, 50, 50))[Accesso Sequenziale (con conflitti)]
      ]
      #import cetz.draw: *
      #cetz.canvas({
        let thread_w = 0.35
        let bank_w = 0.35
        let thread_h = 0.4
        let bank_h = 0.4
        
        // Thread row
        for i in range(32) {
          let x = i * thread_w
          rect((x, 2.5), (x + thread_w, 2.5 + thread_h),
               fill: rgb(220, 200, 200),
               stroke: black)
          content((x + thread_w/2, 2.7),
                  text(size: 5pt, weight: "bold")[T#i])
        }
        
        // Arrows - più thread accedono allo stesso bank (esempio: bank conflict)
        // Threads 0-3 tutti su Bank 0 (rosso)
        for i in range(4) {
          let x_thread = i * thread_w + thread_w/2
          let x_bank = 0 + bank_w/2
          line((x_thread, 2.5), (x_bank, 1.4),
               mark: (end: "stealth"),
               stroke: (paint: rgb(200, 50, 50), thickness: 0.8pt))
        }
        
        // Threads 4-7 tutti su Bank 4 (rosso)
        for i in range(4, 8) {
          let x_thread = i * thread_w + thread_w/2
          let x_bank = 4 * bank_w + bank_w/2
          line((x_thread, 2.5), (x_bank, 1.4),
               mark: (end: "stealth"),
               stroke: (paint: rgb(200, 50, 50), thickness: 0.8pt))
        }
        
        // Threads 8-11 su Bank 8 (arancione)
        for i in range(8, 12) {
          let x_thread = i * thread_w + thread_w/2
          let x_bank = 8 * bank_w + bank_w/2
          line((x_thread, 2.5), (x_bank, 1.4),
               mark: (end: "stealth"),
               stroke: (paint: rgb(255, 150, 50), thickness: 0.8pt))
        }
        
        // Altri thread - pattern simile
        for i in range(12, 16) {
          let x_thread = i * thread_w + thread_w/2
          let x_bank = 12 * bank_w + bank_w/2
          line((x_thread, 2.5), (x_bank, 1.4),
               mark: (end: "stealth"),
               stroke: (paint: rgb(255, 150, 50), thickness: 0.8pt))
        }
        
        // Bank row
        for i in range(32) {
          let x = i * bank_w
          let color = if i == 0 or i == 4 { 
            rgb(255, 150, 150) 
          } else if i == 8 or i == 12 { 
            rgb(255, 200, 150) 
          } else { 
            rgb(150, 200, 255) 
          }
          
          rect((x, 1), (x + bank_w, 1 + bank_h),
               fill: color,
               stroke: black)
          content((x + bank_w/2, 1.2),
                  text(size: 5pt, weight: "bold")[B#i])
        }
        
        // Label
        content((5.6, 0.5), text(size: 8pt, fill: rgb(200, 50, 50))[
          *4 transazioni* - Conflitti serializzano gli accessi
        ])
      })
    ]
  ),
  caption: [
    Differenza tra accesso parallelo e sequenziale alla shared memory.
  ]
) <smem-access-patterns>



#nota()[
  La bandwidth effettiva si riduce di un fattore pari al numero di transizioni separate necessarie.
]

Inoltre possono essere effetuati i seguenti pattern di accesso: 
- Caso *broadcat*: un singolo valore letto da tutti i thread in un warp da un singolo bank (tutti i thread vogliono un certo dato)
``` 
  float f = smem[20];
```

- Caso *parallelo*: un singolo valore letto da un singolo bank. Tutti i thread leggono un dato in posizione diversa 
``` 
  float f = smem[threadIdx.x]
```

- Caso *conflitto doppio*: acceso agli elementi saltandone uno ogni volta (passo 2). Pattern per  gestire strutture di $8$ byte (come `double` o coppie di `float` complesse)
``` 
  float f = smem[(threadIdx.x * 2)%32]
```
  La GPU deve serializzare l'accesso: prima serve il primo gruppo, poi il secondo. Questo dimezza la velocità effettiva della memoria.

- *Nessun conflitto*: accesso agli elementi con un passo di $3$, ad esempio per leggere le coordinate $X, Y, Z$ di un punto: 
```
  float f = smem[(threadIdx.x * 3)%32]
```

#nota()[
  In questo caso *non ci sono conflitti*, in quanto matematicamente, il numero $3$ e il numero $32$ sono coprimi (non hanno divisori comuni). Quando il passo (stride) è dispari e non multiplo di $32$, i thread si distribuiscono perfettamente sui banchi senza conflitti. Tutti i $32$ thread accedono in parallelo, massima velocità.
]

=== SMEM

Se la dimensione non è nota a compile time è possibile dichiarare una variabile adimensionale con la keyword ``` extern```:
- dinamicamente si possono allocare solo array 1D
- può essere sia interna al kernl sia esterna
``` extern __shared__ int array[]```. Alloca un array nelal shared memory.

Per allocare la shared memory dinamicamente la variabile occorre indicare un terzo argomento all'interno della chiamate del kernel:
```c kernel<<grid, block, N*sizeof(int)>>> ```.

#informalmente()[
  Ho dei dati di grandi dimensioni che vengono partizionati in blocchi. Se ragiono a livello di blocco devo pensare che un certo blocco di thread usa dei dati, i dati vengono trasferiti a chunk quando un certo blocco viene scelto per l'esecuzione dall'SM.
]

#esempio()[
  Somma partiziale su blocchi

  Al livello della reduction
  //Inserire esempio

]

=== TILING

Cerco di rendre efficiente la tail 1024 alla volta (32*32). Devo suddividere in tail indipendenti tra di loro, essi andranno combinati tra di loro per ottenere il risultato finale.

#esempio()[
  Prodotto matriciale. I parametei fondamentali sono quelli del risultato finale ovvero righe e colonne.

  Se ogni thread si occupa di una certa cella della matrice risultante (lavora in modo asincrono) se metto in gioco un numero di thread pari alla dimensione della matrice ottengo un thread per ogni cella risultante.

  La shared memory ha senso. Ogni entry (riga) della matrice di input viene usata per molti prodotti.

  #attenzione()[
    Leggere la matrice riga per riga è diversa da leggerla per colonna, in quanto è salavata row major order. Due righe sono contigue ma non è detto che due colonne adiacienti lo siano in global memory.

    la shared memory risolve anche questi problemi di accesso alla memori globale (come sono salvati i dati).
  ]

  La SMEM Ha senso improntala 2D in questo caso. Tappezziamo le due matrici di input (le dividiamo in matrici più piccole), otteniamo dei blocchi disgiunti che coprono le due matrici.  Successivamente all'interno di ogni blocco mi concentrerò su una determinata entry.

  #nota()[
    Se dividessi la matrice in tanti blocchi (massimo 1024 elementi) in casi di matrice di input molto grande creerei un numero troppo elevato di blocchi.
  ]

  //aggiungere immagine
  Strategia. In un blocco lavorano tutti i thread assieme. Il blocco di output conviene usi la shared memory, ma per avere il risultato finale di tutti i thread che appartengono, devo spazoolzare più blocchi delle matrici di input (in base alla lenght e alla height delle matrici di input).

  Il risultato viene calcolato quadrato per quadrato (block size) simultaneamente. Devo avere due SMEM. Per arrivare al risultato dentro il kernel mi serve caricare -> computare -> sincronizzare
]

=== TIling Numba
//aggungere

Convoluzione. Le memorie shard dovrebbero essere anche arricchite con l'alone.
