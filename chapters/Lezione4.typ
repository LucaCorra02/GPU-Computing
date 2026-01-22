#import "../template.typ": *

= Lezione 4

= Parallel Reduction

Operazioni commutative che portano un insieme di elementi (memorizzato ad un array) a un unico elemento

== Somma Array 

Sequenzialmente ha costo $O(n)$ dove $n$ è la dimensione dell'array. 

Per il parallelo (supponendo di farlo in place) le stategie possono essere due: 
- Sommare a coppie. Processo dicotomico (meta della meta ecc)
- fa la stessa cosa ma combina a coppie in maniera diversa. 

Gli schemi in immagine sono uguali sul piano del risultato. Ma a livello  

Stategia parallela ricorsiva: Strategia ricorsiva parallela, la profondità è logaritmica. Ogni step ha un costo che è il numero di elementi che elaboriamo in quello step. 

//aggiungere immagine e capire codice
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

