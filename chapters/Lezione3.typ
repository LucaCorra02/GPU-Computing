#import "../template.typ": *

= Lezione 3

// Da integrare con lezione 2
== Lezione 2 proseguo
Non è corretto calcolare i thread e i blocchi con la divisione intera. è meglio fissare il numero di thread e blocchi in modo fisso. Una volta fissato il numero di thread per blocco divido la struttura dati orginale in blocchi usando una divisione e poi faccio una ``` ceil```.

Funzioni ``` device```. Sono funzioni utiizzate internamente al kernel, lavorano esclusivamente sulla GPU. Devono essere marcate con il decoratore ```py @cuda.jit(device=True)```. Inoltre possono avere tipo di ritorno a differenza del kernel. 

#attenzione()[
  Le funzioni device non son un kernel
]

Impostando l'opzione ``` inline = True``` il codice della funzione device viene inserito dal compilatore dentro il kernel, risparmiando il costo della chiamata. 

Di default il trasferimento avviene in maniera automatica, (potremo creare una struttura numpy e passarla indipendentemente a host e GPU). Tuttavia è sempre meglio essere conservativi. Risolve problemi:
- trasferimenti pesanti
- consuma banda
- aumenta la latenza (esecuzione che aspetta i trasferimenti)

== Pattern di accesso alla memoria

Panoramica. Ci sono dei trhead collezionati in blocchi i quali hanno la possibilità di avvalersi di una memoria shared. I thread rispetto a tutto ciò che non è nel blocco hanno uno statuto speciale che permette di gestire la memoria. 
#attenzione()[
  è necessario avere delle primitive di sincronizzazione per cooperare. 
]
Le griglie sono più blocchi di thread. Esse le associamo a un compito finito (programma).

=== Mapping logico fisico

un blocco di trhead arriva ad un SM e un thread viene mappato su un core fisico. Dobbiamo capire come le grid si spalmano sul device

== SIMT

Single istruction multiple thread 

*Warp* = L'idea è che abbiamo una trama di thread che eseguno la stessa istruzione (codice unico dettato dal kernel) su dati diverse. Un warp è costituito da 32 thread (con ID consecutivi). 

il warp è una costante anche su varie architetture, in modo tale da garantire l'interoabilità. Idealmente tutti i thread in un warp eseguono in parallelo allo stesso tempo (modello SIMD), non c'è concorrenza all'interno dello stesso warp, Idealmente tutti i thread evolvono in parallelo su dati diversi. 

Il modello SIMT (non ideale) introduce un PC per ogni thread. 

L'hardware se ne sbatte dell'organizzazione a blocchi dei thread, essa è solo un organizzazione logica. L'hardware vede la griglia come sequenze di warp. 

i blocchi devono essere multipli di 32 in modo che si mappano bene sull'hardware (anche se la perdità di prestazioni non è alta).  

Se ho un blocco di 128 thread avro 4 warp da 32 thread. L'id dei trhead nel warp è sequenziale. 

L'ordine 2D("logico") viene linearizzato in 32 thread sequenziali. 

Lo scheduler va a schedulare i warp per l'esecuzione fisica. 

Passaggi: 
- i blocchi vengono schedulati sull SM
- i warp di cui è costituito un blocco vengono schedulati e succissivamente eseguiti (servono 32 core all'interno di un SM per eseguire un warp), la perfezione è che avviene tutti in modo sincrono. Anche le memorie sono improntante a dimensione 32. Questa è la situazione che da il massimo througtput generale. 

Ci sono ragioni in cui un WARP può essere in wating in quanto tutte le sue 32 linee non sono pronte. 

//riguardare
Limitazione dello scheduling:
- Meno warp presenti nella lista dei pronti
- più latenza nell'accedere ai dati
- se saturo il numero di risorse per un thread singolo (tante varibili saturo i registri) lo sheduling genrale soffre. 

=== scheduling dei blocchi

la griglia è una collezione di blocchi (non c'è distinzione, noi la facciamo a livello logico), ogni blocco viene associato ad un SM. 

Gli SM consumano memoria condivisa. Il numero di blocchi può essere arbitrario (anche se limitato nel numero di thread), inoltre ogni blocco è indipendente nello scheduling. Ogni blocco lavora in locale in base alle risorse dell'SM. 

C'è un meccanismo di cooperazione tra blocchi si estende la sincronizzazione tra blocchi. 

=== Scheduling dei warp

ci possono essere più warp attivi all'interno del device. i warp vengono schedulati e eseguiti per istruzione per istruzione. 

i warp possono sincronizzarsi all'accesso in memoria attraverso delle primitive. 

Obbiettivo :
- ridurre la latenza avendo il numero massimo di warp attivi contemporaneamente.
$
  "Active warp" / "max possible warp per SM"
$

I branch rendono inefficiente il sistema. Un brench avviene nel codice, un thread ID segue una strda e un altro thread ID un'altra. é possibile riorganizzare i dati a livello di warp e non di thread. 

#attenzione()[
  Non è detto che c'è un assocazione 1:1 thread e dati, ovvero il primo dato corrisponde al thread con ID 1. Ma posso lavorare modulo la dimensione di warp 
]

=== Cluster

Possiamo associare più bloccho ad un cluster. I blocchi nello stesso cluster possono cooperare tra di loro. Aggungiamo un overhead di gestione per permette di gestire strutture dati più grandi. 

Nella situazione ci sono dei gruppi fisici di SM (GPC) che permettono questa cooperazione

=== Divergenza ed esecuzione

//Aggiugnere immagine ed esempio
i branch nel codice  spezzano i thread del warp in due. I thread non sono più eseguiti in modo parallelo ma in modo sequenziale. Di 32 linee paralle divido in due plotoni da 16

i branch vanno a inficiare sui warp, posso organizzare il codice per far si che non accada a livello di warp. 

//Aggiungere esempio
#esempio()[
  Spalma in due gruppi i warp. i branch *non hanno un impatto banale*. 

  Situazione ideale se avessi un array di 64 dovrei dire che i primi 32 si occupano dei pari e gli altri dei dispari. Posso sempre ragionare a gruppi di 32.  
]

=== Sincronizzazione

//aggiungere immagine
#nota()[
  Sincronizzazione livello di blocco principalmente (anche se esite per il warp )
]

Nel modello SIMT ogni thread può fare strade diverse. Ogni thread ha un flusso e può richiedere tempi diversi. Richiede sincronizzazione, quando si riparte con la prossima istruzione dobbiamo essere sicuri che i thread siano tutti allo stesso punto. 

in C esistono una serie di primitive: 
- ``` syncthreads``` = a un certo punto del codice kernel compare questa istruzione (interpretata da runtime CUDA). intriduce una barriera per i thread del blocco 
- ``` syncwarp``` = introduce sincronizzazione a livello di warp. 

//aggiungere immagine
Ogni thread ha un proprio progam counter PC (ognuno ha un registro). Di conseguenza i thread possono essere gestiti con divergenza e ri-convergere anche a livello di warp.  (poco interessante per il prof?)















