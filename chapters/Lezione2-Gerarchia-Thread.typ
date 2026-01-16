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

=== Dati lineari (1D)





=== Dati piani (2D)

=== Dati volumetrici (3D)







Esempio di griglia 1d, con blocchi 1d: `Grid(3,1,1) <-> grid(x,y,z)`. Si ragiona in uno spazio cartesiano.\
In questo caso ho 3 blocchi (id da 0 a 2). Ogni blocco ha 4 thread `block(4,1,1)`. 

Nel secondo esempio ho sempre una griglia da 3 blocchi, tuttavia ogni blocco è definito da `block(3,3,1)`. Esempio adatto per strutture a vettore. 

Esempio di griglia `2D` e blocchi `2D`. In questo caso ho 1 griglia su due livelli, ogni livello contiene 3 blocchi da $3*3$ thread. Questo esempio serve per mappare un thread univoco ad ogni cella di una matrice (nostro dato)

*Nota* := Un blocco può contenere al più 1024 thread. Di conseguenza devo fare la seguente operazioni, numero di thread che mi servono modulo la dimensione del blocco. 










== Google colab in Vs Code

Solitamente si usa la programmazione strutturata base per definire le funzioni parallele. 

Solitamente si cerca di evitare le allocazioni dinamiche di oggetti in python dentro al kernel.

Il codice viene sempre scritto lato host, vengono scritte delle funzioni cuda che devono essere eseguite sulla CPU. 

== Gerarchia dei thread

=== Modello Hardware
Ogni SM ha la propria memoria (chache e registri). Ci sono anche delle organizzazioni logiche degli SM. 

=== Thread in cuda

Dall'efficienza dello scheduling dipende quanta risorsa chiediamo per ogni singolo thread. Se il blocco di thread è molto esoso in termini di risorse può essere l'unico ad essere eseguito in un certo istante -> va gestito questo problema. 

i thread a volte devono coperare nella GPU. Sincronizzazione all'interno di blocchi logici di thread. 

Controllo: la gestione dei thread e della memoria dati è nelle mani del programmatore. 

Le operazioni di lancio del kernel sono asincroni, mentre i trasferimenti di memoria sono sincroni (per mantenere la consistenza dei dati). 

=== Processing FLow
- Input copiato da CPU a GPU 
- Caricare il programma da eseguire su GPU ed eseguirlo
- Intercettare i risultati dell'elaborazione

=== Gerarchia dei thread

Gerarchia su due livelli (astratti). Ovvero:
- Griglia = collezione di blocchi ordinata
- blocchi = collezione ordinata di thread

Ciascuno di questi due ordini di astrazione può essere gestito in 3 dimensioni. Spesso lo si fa corrispondere a dei dati, riconducibili a questi ordini. 

Esempio di griglia 1d, con blocchi 1d: `Grid(3,1,1) <-> grid(x,y,z)`. Si ragiona in uno spazio cartesiano.\
In questo caso ho 3 blocchi (id da 0 a 2). Ogni blocco ha 4 thread `block(4,1,1)`. 

Nel secondo esempio ho sempre una griglia da 3 blocchi, tuttavia ogni blocco è definito da `block(3,3,1)`. Esempio adatto per strutture a vettore. 

Esempio di griglia `2D` e blocchi `2D`. In questo caso ho 1 griglia su due livelli, ogni livello contiene 3 blocchi da $3*3$ thread. Questo esempio serve per mappare un thread univoco ad ogni cella di una matrice (nostro dato)

*Nota* := Un blocco può contenere al più 1024 thread. Di conseguenza devo fare la seguente operazioni, numero di thread che mi servono modulo la dimensione del blocco. 

==== Blocco di Thread

Un blocco di thread è un gruppo di thread che possono coperare tra loro mediante: 
- Block-local synchronization
- Block-local shared memory

i thread di differenti blocchi possono coperare come cooperative groups. Inoltre tutti i thread di una certa grid condividono lo stesso spazio di global memory (tuti i thread condividono la stessa memori ). Ad ogni thread è associato : 
- BlockIdx (indice del blocco nelal griglia)
- ThreadIDx
Gli indici stanno in variabili build-time popolate a runtime (si tratta di un tipo specifico). 

Le dimensioni di grid e block sono specidicae dalle seguenti variabili : 


`dim3` = Si tratta di un header in c. 
Riguardare

Per indicizzare un blocco 2d, dobbiamo prima indicizzare la colonna $x$, accediamo ad una cella usando la riga come offset $x+y*D_x$.

=== Grid-1d Block-1d
`int idx = blockDim.x * BlockIdx + threadIdx.x` riesco a ispezionare tutti gli id unici dei thread della griglia (aggiungere immagine).

Somma di vettori. Usiamo più blocchi. il vettore di input è $1024*1024$. In questo caso abbiamo $32$ htread per blocco, il numero di blocchi è $1024$ blocchi da $32$ thread copro tutto lo spazio. 

I kernel vengono smistati nella GPU, se sono tatni ma piccoli vengono eseguiti assieme se ne arriva una grande prende tutta la GPU. 

L'host può lanciare solo funzioni etichettate come global. Se la funzione è device può essere lanciata solamente dal kernel. 

=== Allocazione dinamica della memoria

In c l'allocazione della memoria avviene tramite malloc. In c organizza i dati di un array multidimensionali in row-major order (linearizzati). Elementi consecutivi delle righe sono contigui. 
Matrice $m x n$ (m righe e $n$ colonne)

Per calcolare $"idx" = i*n+j$ $j$ è l'offset e dobbiamo scavalcare $n$ volte. Accedo così al chunk linearizzato in memoria. 

Nel caso di una grid 2d (guardare immagine colorata). 

Somma di matrici, prendo le x e le y per colonne e righe e vado a costruire un indice rinealizzato che raggiunge tutti gli elementi della matrice. 