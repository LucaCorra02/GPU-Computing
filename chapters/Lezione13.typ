#import "../template.typ": *

Un grafo può essere rappresentato come una matrice (?).
Ad ogni nodo può essere associata un informazione per modellare più situazioni basate sui grafi.

#esempio()[
  La molecola può essere vista come un grafo, ogni nodo ha un significato associato
]
Potrei anche non conentrarmi sulla struttura del grafo in toto ma solo sul significato associato al nodo o all'arco.

== Immagine come grafo

Attraverso le CNN. Le immagini possono essere dei casi speciali di grafi. Possiamo vedere i nodi come pixel, gli edge sono i primi vicini pixel (nello spazio euclideo).

Un kernel è formato dal vicinato, analizza i pixel vicini ad uno. Nel caso di grafo il vicinato può essere più irregolare, non abbiamo una griglia ben definita.

= CNN

Le reti neulari convuluzionali hanno le seguenti proprieta:
- Il kernel è una maschera condivisa da tutti i pixel, essa viene appresa

*Equivarianza*: proprietà che deve avere un modello. Se prendo una funzione è una trasformazione dei dati mi devo chiedere come il modello risponde a delle trasformazioni (sensibilità del modello). Ad esempio nelle immagini gli oggetti possono avere degli shift (posizione relativa di un oggetto), se la CNN non fosse robusta a questa proprietà è un problema.

Data una funzione $f$ che è la CNN prendo una trasformazione $T$ e se applico la triasformazione all'ingresso $F(T x)$ deve essere robusta, ovvero deve darmi il trasformato dell'applicazione di funzione $T(f(x))$

#esempio()[
  Se ho trovato un cane in un immagine e applico uno shift, alll'immagine iniziale devo comunque ritrovare il cane, robustezza.
]

Una funzione $G$ è *invariante* per una trasformazine $T$ se:
$g(T x)$ = g(x)

In questo caso utile per la classificazione. Voglio un immagine tale per cui applico lo shift all'immagine e voglio predirre la stessa label.


Nell'ambito dei grafi vogliamo modellare  dei modelli modellati a grafo che permettono di costruire i seguenti task:
- Prorpietà a livello di nodo, predizione dei nodi o archi
- Predizioni rispetto al grafo complessico.

#esempio()[
  Node prediction. Vogliamo infierire le label per i nodi. Ad esempio dei documenti con delle cross-reference e vogliamo capire i documenti correlati tra di loro. Meccanismi di raccomandazione


  edge prediction ci permette di predirre le proprietà tra due nodi, ad esempio proprietà tra delle proteine.

  Predizioni a livello di grafo, ad esempio la predeizone dell'energia dello stato del sistema.
]
Quando guardiamo l'intero grafo, le task di classificazione possono tendere alla regressione.

== Approccio induttivo
Un dataset è dato da un insieme di grafi, il modello vuole prendere contesto, informazioni generali ecc e vogliamo rispondere in modo generale (da un campione devo essere in grado di generalizzare).

== Approccio trasduttivo
Dato un grafo (grafo del web unico) vogliamo arrivare a dedurre delle proprietà di qualche nodo.

Si tratta dell'approcio semi-supervisionato.

== Graph rappresentation learning

Vogliamo applicare una serie di trasformazioni (bias induttivi, trasformazioni di mebedding) che ci permettono di acquisire una conoscenza globale del contesto (ad esempio i Transformer sono capaci di capire il contesto).

In questo caso i nodi del grafo si scambiano informazioni (hop) fino a costrutire una conoscenza generale. Da qui si deinisce un task specifico, i nodi finali saranno più ricchi di conoscenza.

=== Node, Edge and graph embedding

Una volta che ho acquisito un modello generale posso fare find-tuning su dei grafi più particolari.

//riguardare

Ci sono delle analogie con i tranformer:
- Lavorano sul contesto: CNN parte da contesto vicini e mano a mano si espande
- Posso affrontare diversi problemi, in contesti in cui l'inputè strutturato a grafo. le task sono:
  - Regressione
  - Classificazione

== Grafo definizione

Un grafo è un insieme dei nodi e archi. Il vicinato del nodo $n$ è il vicinato $N$ ovvero i nodi collegati a $n$.

La matrice di adiacienza. Avende $n$ nodi e $m$ archi è una matrice $n times m$. Un grafo non orientato vale la simmetrica

//aggiungere matrice
Nella prima ho un ordinamento dei nodi $A,B;C;D;E$ nella seconda ho un ordinamento dei nodi diverso. Ho due matrici diverse anche se il grafo è diverso.

Se le reti CNN sarebbero sensibili al riordinamento dei nodi (a parità di grafo) sarebbero inutili.

Le matrici di adiacienza è una rappresentazione problematica, viene dato implicitamente un oridinamento dei nodi. Il grafo sottostante non cambia, non si può usare $A$ direttamente.

Vogliamo codificare l'invarianza e le permutazioni come uno degli espetti fondamentali. Vogliamo delle reti che siano invarianti e equivarianti //riguardare differenza.

== Permutazione di nodi
Una matrice di permutazione è una matrice quadrata di dim $m times m$ in cui ho un singolo valore per riga e per colonna gli latri sono a zero (matrice di identità scambiata)

//aggiungere immagine

La matrice prende l'ordinamento $A,B,C,D,E$ e ottengo una permutazione. Ho dei vettori unitari (una componente 1 e le altri a zero) e applico una permutazione $pi$ che permette una permutazione delle righe e colonne della matrice di identità.

é la trasformazione $pi$ che caraterizza la matrice $P$ permutazione della matrice di identità. Le possibili permutazione sono $n!$

Effetti della permutazione:
- Data una matrice delle feature $X in R^(N times D)$
- Facico una permutazione con $P$, potrei avere riordinamento dei nodi:
  $
    hat(X) = P X
  $

Gli efffetti della permutazione degli input si traduce in una permutazione della matrice di adiacienza nel seguente modo:
$
  hat(A) = P A P^T
$
//aggiungere parte matematica
#dimostrazione()[
  Data una matrice $P_N times N$. Somma righe j = 1 e  somma colonne i = 1.

  Date le feature associate ai nodi del grafo. $X^(R times D)$. Una matrice di permutazione secondo $pi$ trasforma una matrice X in $tilde(X)$:
  $
    tilde(X) = P X
  $
  //aggiungere esempio e continuare
  Data la matrice $A$ se permuto $X$ per ottenere la matrice $tilde(A)$ associata a quella permutazione $P$ devo fare:
  $
    P A P^T
  $
]

== Message Passing in GPU-Computing

Possiamo vedere la CNN come qualcosa che mette in relazione i nodi e il vicinato. Il nodi $i$ ha un certo vicinato. Le GNN devono essere in grado di prendere le informazioni dal vicinato e trasferire delle informazioni al nodo centrale che usareara in qualche modo.

Se prendo un filtro $3 times 3$ al livello $l+1$ definisce una funzione:
$
  z_i(l+1) = sigma(sum_j w_j z_j^(l) + b)
$
La CNN spazzola tutte le patch e i $9$ valori dei pazi sono appresi da tutti i pixel. Funziona per le patch ed  il liite delle CNN. Non è equivariante ai $w_j$ se applico una permutazione alle label dei pixel non è equivariante.

Passo a qualcosa che non è equivariante rispetto alla maschera dei 9 pixel, ma vado a pensare ad un peso $w$ che non dipende dal riordino dei vicini. Il peso è unico e viene moltiplicato dagli input che vengono dal vicinato.

//aggiungere formula

Questo meccanismo prende il nome di *message passing*. L'equazione può essere divisa in due parti:
- raccolgo le informazioni dei vicini e le faccio a recapitarel nodo i
- le mando al nodo i il quale li combina

La rete condividendo i due pesi e il bias dei nodi del vicinato divente una rete equivariante.

= GCN

Il message passing è simile alla convulazione. Abbiam un modello:
- Differenziabile, usa la backpropagation
- Paranetri a livello
- il livello n al nodo l è un embedding D-dimensionale associato a quel nodo.

Per poter fare questo dobbiamo definire un modello di:
- aggregazione info dai vicine
- calcolo delle informazioni a livello precedente

//aggiungere immagine
#esempio()[
  L'embeddinf a livello k +1 subisce una update a livello k dato da se stesso al livello precedente e tutto il vicinato del nodo v sempre a lviello precedente.

  Questo per ogni livello.
]

== Aggregazione

//aggiungere formula

Le proprietà di aggregazione:
- vale per un vicinato non fissato (vari rispetto al grafo)
- invariante rispetto alla permutazione
- include dei parametri che devono essere appresi

=== Aggregazione somma

Forma semplice, somma: //aggiungere formula

La somma può crescere motlo, può essere soggetta a normalizzazione.

=== Media somma

L'intero grafo ha un parametro che è il cammino più lungo (diametro del grafo). Il $K$ è il campo recettivo, indica quanto esploro del grafo, se arrivo al campo del grafo ho fatto recapitare tutte le informazioni a tutti i nodi.

Ha delle propietà spiecevoli, over smoothing, non è utile alzare il $k$ (numero di layer della GNN). Solitamente il $k$ varia tra $3$ e $5$. Il numero di hop solitamente non viene alzato di molto.

== Update step

Considera se stesso al livello $L$ e calcola l'aggregato dei vicini.
//aggiugnere forma

== Aggregazione learnable

costruzione dell'mebedding a livello $n$. I prametri devono essere apprendibili. Per questo motivo usano degli MLP:
$//aggiugnere formula
$
L'aggregazion sfrutta due MLP. Viene operata una trasformazione condivisa a tutti i vicini (stesso peso) e tutto questo viene appreso per la funzione finale. Dove:
- $"MLP"_theta$
- $"MLP"_j$
condividono i pesi di su tutti i nodi e sono universali (un qualsiasi MLP, transformer ecc).

=== Layer

Tipicamente $h_n^0$ è l'input.

Nelle elevate dimensioni si arriva ad avere delle trasformazioni:
//aggiunger formula

== Update non learnable

//riguardare

= PyTorch Geometric

Le api principali sono `GCN conv` e `Graph conv`. Hanno dei meccanismi di differenziazione automatica e parametri learnable.

== GCN conv

Introduce una normalizzazione di Kipf and Welling. Il sistema è più robusto e sensibile ad outliers.

Matrice la placian(?) //riguarda e aggiungere matematica
Mette insieme i gradi dei nodi. Dal punto di vista dei gradi un grafo può essere molto sbilanciato, va da zero a n.

//guardare su notebook.

Se prendo $D^-1/2$ prendo la diagonale  e ne faccio 1 sulla radice.

== Modello Saige

impara anche la forma di aggregazione e non direttamente gli embedding. La formula non è predefinita ma viene apresa, attraverso la concatenazione.

è diverso dalle GCN perch


#esempio()[
  Problema di classificazione. Feature bag of word, uno zero a secondo della presenza assenza della parola. I nodi sono le pubblicazioni e gli edge le citazioni.

  Su ogni nodo c'è la feature (paper). La classificazone sono 7 topic di ricerca.

  Il grafico utilizzato è uno.

  `GCN_leyer` è una poezione che deve essere appresa. Si tratta di una matrice. Il vicinato è ricavato dalla matrice di adiacenza e dalle marice dei nodi. I vicini vengono agregati solamente con il prodotto della matrice.

  La rete GCN sono due livelli del GCN layer. Impara le feature, una feature per ogni nodo è la predizione.

  nell'aprendimento facciamo il maschermaneto, validiamo con la trianing mask, validadiamo con la vaidate mask e il test con la test mask.
]
// Guardare la cosa dei paper.
