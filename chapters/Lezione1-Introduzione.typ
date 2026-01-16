#import "../template.typ": *

= Introduzione

== Perchè HPC? 
L'obbiettivo è svolgere più operazioni possibili nell'unita di tempo.
- EXA = numero di operazioni floating point nell'unità di tempo
CPU = host, GPU = device, ad oggi sono collegate assime per abbatere il tempo della trasmissione (collo di bottiglia negli HPC). 

Legge di Moore:= mantenuta in vita dalle gpu. La cpu ha invece raggiunto delle barriere ormai insuperabili. 

Nasce il paradigma Gp-GPU. La cpu è l'elemento dominante nell'orchestrazione del sistema. La gpu è il servente della cpu (unità di calcolo).

== Motivazioni per il calcolo parallelo

Supponiamo di considerare il calcolo tra due matrici, il costo è $O(n^3)$ in un sistema di calcolo sequenziale. Siccome il risultato di ogni cella della matrice è indipendente, possiamo andare ad applicare il parallelismo (spinto). Calcoli alla base di un qualsiasi modello di AI, da qui la necessità di renderli efficenti. 

CCN = Reti che fanno lo stesso calcolo molte volte (tecnologia ormai superata), ad esempio operazioni che devono essere fatter per ogni pixel. 

==== Parallelismo
Capacità di eseguire parti di un calcolo in modo concorrente (non camnbia la semantica del calcolo, solo il modo in cui viene effettuato). Non è detto che sia tutti paralelizzabile (magari solo delle parti).

Con parallelismo si può:
- risolvere problemi più grandi nello stesso tempo
- risolvere un problema di dimensioni fisse in un tempo più breve

Grado di parallelismo = quanti sono grandi le unità perallele. Noi usermo i thread, ovvero unità di esecuzione costituita da una sequenza di istruzioni e gestita dal sistema operativo ( o da un sistema runtime, sincronizzazione), concorrono nell'accesso alle risorse

=== Paradigma GP-GPU
- General purpose = uso della gpu per eseguire computazioni 
- Modello eterogeneo = uso di CPU (host) e GPU (device) assieme per dare vita a un modello di calcolo
- Separazione = Gestione affidata al compilatore (per la parte sulla CPU)

La GPU necessità di trasferimenti diretti di memoria da parte della CPU. La CPU gestisce la sincronizzazione. 
C'è una differenza di esecuzione dei task: 
- CPU 
- GPU = architettura massiciamente parallela

il codice prodotto è costituito da due parti sia cpu che gpu, bisonga esplicitare cosa va su cpu e cosa su gpu, ognuna sarà specializzata nell'esecuzione di determinati task. I punti di sincronizzazione sono fondamentali per cordinare i due sistemi di calcolo. 

il calcolo parallelo può essere realizzato in vari modi : 
- parallelismo dei dati = suddivisione dei dati in parti uguali e elaborazione simultanea di ogni parte su più processori (moltiplicazione matriciale)
- parallelismo di task = suddivisione del lavoro in attività indipendenti e l'assegnazione simultanea di queste attività a diversi processori 
- parallelismo di istruzioni = suddivisione delle istruzioni di un programma in parti indipendenti e l'esecuzione simultanea di queste istruzioni su più processori

Non tutti i dati sono adatti al parallelismo, dipende dalla dimensione e dal tipo. Il modello di calcolo più adatto varia in base al tipo del problema. 

=== Parallel computing 

aggiungere

Modello PRAM: 
- Modello a memoria condivisa o distribuita, processori indipendenti ma sincroni (sincronizzati tra di loro). 
- .....

Tipi di istruzioni, tassonomia di FLynn
- SISD (single-istruction single-data)
  - Modello della CPU tradizionale
  - No parallelismo, un'istruzione alla volta
- SIMD (single-istruction multiple-data)
  - Computazione vettoriale
  - Unica istruzione su più dati. Ciascuna unità di calcolo lavora e accede a dati diversi
- MIMD (moltecipli istruzioni molteplici dati)

=== SIMT (Nvidia)
Oltre la tassonomia classica
  - Ogni thread può seguire dei flussi di istruzioni distinti dagli altri. Eseguono lo stesso codice ma nei punti di branch possono prendere strade diverse (devono essere applicati a dati distinti)
  - Serve hardware adeguato

Caratteristiche : 
- ogni thread ha il proprio instruction address counter
- ogni thread ha il proprio register stare e in generale un register set
- ogni thread può avere un execution set indipendente

== Nvidia GPU

=== SM

=== Tensor Core
In un colpo solo viene fatto un prodotto matriciale tra matrici piccoline (una matrice grande viene divisa in sotto-matrici), il calcolo avviene in parallelo. Serve dell'hardware dedicado, se non avessi dei cuda core dovrei utilizzare più core classici. 

Nvidia tesla T4 (la useremo in colab)

MIG architettura

=== CUDA 
Framekork per Nvidia per il calcolo parallelo. Abbiamo un sacco di librerie, tra cui _pyTorch_. 

Compilando un codice CUDA _.cu_ usiamo il compilatore _NVCC_:
- Prende in ingresso tutto e produce del fat-binary ovvero che comprende sia la parte di host che device. 
- JIT = il codice viene ottimizzato dal compilatore. Si tratta di un assembly intermedio che viene prodotto. 

Qualificatore global, l'esecuzione della funzione etichettata viene demandata alla GPU. _threadIdx_ è l'identificatore unico del thread.

$<<<1,10>>>$, esprime la configurazione del kernel, in questo caso la GPU parte con 10 thread che eseguono tutti lo stesso kernell. 

CudaMalloc = allocazione sul device atti a ospitare quello che ho sull'host. 

cudaMemcpy = Ci permette di trasferire i dati, da host a device. 

Passaggi : 
- Trasferisco su device
- Eseguo somma su device
- Trasferisco 

== Numba CUDA 

Operazioni su GPU tramite python (non tutto ma solo delle primitive).

C'è uno stretto legame tra _NUMBA_ e _Numpy_, forte legame. 

Si usano i *decoratori*, ovvero funzioni che modificano altre funzioni, definiti dalla sintassi 
  @ Decorator 

Il decorator prende una funzione come parametro, solitamente fa prima e dopo. Numba prende il decoratore e fa cose con il compilato. 

Guardare slide per questa parte. 

`@python
  from numba import cuda
  @cuda.jit
  def kernel(a):
    pass
`
se antepongo il decoratore jit a una funziona, la funzione diventa un kernel.

Una volta definiti i kernell, li lanciamo con l'opportuno decoratore, specificando il numero di thread il numero di blocchi e gli argomenti. 

Se non mettessi `cuda.synchronyze()` quando l'interprete (python interpretato) lancia il kernell, non si curerebbe affatto di aspettare il risultato ottenuto dal calcolo parallelo. 

=== Somma parallela di vettori

vedere da slide

il default di python è 64 se definisco un intero. In GPU significa invocare delle specifiche unità di calcolo dei 64 che se non ci sono vengono simultate dai 32 (molto più lento). Bisogna ossessivi nel specificare sempre il tipo. 

I kernell come in C non restiutiscono valore. Avviene tutto per riferimento. 

Il kernell è sempre asincrono, l'istruzione successiva al kernell viene eseguita dalla CPU. I trasferimenti di memoria invece sono implicitamente sincroni. 