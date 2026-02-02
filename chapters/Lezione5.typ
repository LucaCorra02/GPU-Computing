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

=== Parallel access via banking

La memoria shared viene divisa in $32$ moduli della stessa ampiezza chiamati *bank*. La dimensione dei banchi dipende dalla dimensione dei dati, solitamente $4$ byte(int), ovvero una word. I banchi solitamente implementato sequenze di word.

#nota()[
  La shared memory è tagliata sulla dimensione del warp, in modo tale da aumentare l'efficienza.
]

Ogni richiesta di accesso fatta di $N$ indirizzi che riguardino $N$ distinti bank sono serviti simultaneamente. 

#nota()[
  Ogni SM ha una quantità limitata di shared memory che viene ripartita tra i blocchi di thread. Il numero massimo di blocchi ospitati all'interno di un SM può scendere se all'interno di un blocco vengono richieste molte risorse (registri e shared memory), in quanto esse sono condivise nell'SM. Il numero massimo di blocchi non è mai fisso ma dipende dall'*occupancy* (tasso di warp attivi simltuanamente nell'SM).
]

Pattern di accesso. Possiamo avere: 
- Acesso parallelo, ogni thread accede ad un blocco differente
- Accesso sequenziale, si possono creare dei conflitti se diversi thred cercano di accedere allo stesso blocco. 

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
