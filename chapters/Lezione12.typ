#import "../template.typ": *

= Transformer

I *Transformer* sono una delle architetture più importanti nel deep learning moderno, introdotti recentemente nel $2017$.

#informalmente()[
  Un Transformer prende un insieme di vettori in uno spazio di rappresentazione e li *trasforma* in un altro insieme di vettori:
  - Stesso numero di vettori
  - Stessa dimensionalità
  - Nuovo spazio di rappresentazione più ricco

  L'obiettivo è produrre una rappresentazione interna più espressiva, adatta a diversi compiti (classificazione, generazione, ecc.).
]

Essi si basano su un meccanismo fondamentale: *l'attenzione*. Tale oggetto, permette al modello di decidere su cosa concentrarsi. Non tutte le parti di un input sono ugualmente rilevanti. L'attenzione darà *pesi diversi* a diverse parti dell'input, permettendo al modello di focalizzarsi sulle informazioni più importanti per il compito.

Le pricnipali caratteristiche dei Transformer sono:
/ Scalabilità: i Transformer si mappano efficientemente su hardware parallelo (GPU). Modelli con $10^12$ parametri mostrano *capacità emergenti*, talvolta descritte come primi segnali verso l'intelligenza artificiale generale (AGI).

/ Versatilità: le architetture Transformer sono state estese con successo a:
  - Testo (NLP)
  - Immagini (Vision Transformer, ViT)
  - Audio, video, DNA
  - Modelli multimodali

/ Apprendimento auto-supervisionato: i Transformer possono essere addestrati su dati non etichettati (es. testo grezzo da Internet), sfruttando la *scaling hypothesis*: aumentando la dimensione del modello e dei dati, le prestazioni continuano a migliorare.

#nota()[
  Prima dei Transformer, i modelli di NLP più diffusi erano:
  - *Bag of Words (BoW)*: ignora l'ordine delle parole, non può esprimere il contesto. In questa tecnica abbiamo un vocabolario fissato. Ogni parola è rappresentata da un indice intero, e una frase è rappresentata da un vettore di conteggio (o frequenza) delle parole presenti. Non cattura la struttura sintattica o semantica del testo.

  - *Reti Neurali Ricorrenti (RNN)*: processano il testo sequenzialmente, difficili da parallelizzare e soggette a dimenticare informazioni a lunga distanza.

  I Transformer superano tutti questi limiti grazie alla self-attention.
]

== Testo -> Embedding

Una parte fondamentale di qualsiasi modello di NLP è la *rappresentazione del testo*. I Transformer non fanno eccezione. Il testo viene convertito in vettori densi (embedding) che catturano il significato semantico.

I tranformer come prima cosa, presentano una fase di *tokenizzazione* e *embedding*:
+ *Tokenizzazione*: il testo viene suddiviso in token (parole, sottoparole o caratteri).
+ *Vocabulary mapping*: ogni token riceve un identificatore intero univoco. La frase originale viene convertita in una sequenza di ID:
  $
    x_1, x_2, dots, x_N in {0, dots, K-1}
  $
+ *Embedding layer*: ogni ID viene mappato a un vettore denso tramite una matrice $W$ appresa:
  $
    W in RR^(K times D)
  $
  dove $K$ è la dimensione del vocabolario e $D$ è la dimensione dell'embedding (features). Viene eseguita una semplice operazione di *lookup*, dove l'ID del token viene usato come indice per estrarre il corrispondente vettore di embedding dalla matrice $W$:

  L'*output* è una lista di vettori densi:
  $
    x_1, x_2, dots, x_N in RR^D
  $

#esempio()[
  Consideriamo un vocabolario di $K=5$ token e embedding di dimensione $D=4$. La matrice di embedding è:

  #align(center)[
    #cetz.canvas({
      import cetz.draw: *
      
      // Matrice W
      let cell-width = 0.8
      let cell-height = 0.6
      
      // Disegna la matrice W
      for i in range(5) {
        for j in range(4) {
          let x = j * cell-width
          let y = -i * cell-height
          rect((x, y), (x + cell-width, y + cell-height), stroke: black)
          content((x + cell-width/2, y + cell-height/2), text(size: 9pt, $w_(#i #j)$))
        }
      }
      
      // Label W
      content((-0.6, -1.2), $mb(W) =$)
      
      // Dimensioni
      content((1.6, 1), text(size: 9pt, $D=4$))
      content((-0.6, -2.3), text(size: 9pt, $K=5$))
      
      // Freccia e lookup
      line((3.8, -1.2), (5.0, -1.2), mark: (end: "stealth"))
      content((4.4, -0.9), text(size: 8pt, [lookup]))
      
      // Token ID
      content((5.5, -1.2), text(size: 10pt, [ID = 2]))
      
      // Freccia verso vettore risultante
      line((6.2, -1.2), (7.4, -1.2), mark: (end: "stealth"))
      
      // Vettore risultante (riga 2 della matrice)
      for j in range(4) {
        let x = 7.6 + j * cell-width
        let y = -1.2 + cell-height/2
        rect((x, y - cell-height/2), (x + cell-width, y + cell-height/2), 
             stroke: black, fill: rgb("#e8f4f8"))
        content((x + cell-width/2, y), text(size: 9pt, $w_(2 #j)$))
      }
      
      // Label vettore
      content((9.5, -0.2), text(size: 9pt, $mb(y) in RR^4$))
    })
  ]

  Dato un token con ID = 2, l'embedding corrispondente è semplicemente la *terza riga* della matrice (indice 2):
  $
    mb(y) = [w_(2 0), w_(2 1), w_(2 2), w_(2 3)]^T
  $

  Questa operazione è estremamente efficiente: non richiede moltiplicazioni, solo un accesso indicizzato alla memoria.
]


== Self-Attention

#attenzione()[
  Un embedding *statico* è insufficiente: la parola "banca" ha significati diversi in contesti diversi ("banca del fiume" vs "conto in banca"). La *self-attention* permette a ogni token di aggiornare la propria rappresentazione guardando tutti gli altri token nella sequenza.
]

L'attenzione è un meccanismo che permette al modello di combinare vettori di input con *pesi dipendenti dai dati*.

I dati vengono processati come un *insieme di vettori* (token), ognuno con $D$ features. Data una sequenza di $N$ vettori token ($N$ = dimensione del contesto):
$
  x_1, dots, x_N in RR^D
$
La matrice dei token è:
$
  mb(X) = mat(
    x_(1,1), x_(1,2), dots.h, x_(1,D);
    x_(2,1), x_(2,2), dots.h, x_(2,D);
    dots.v, dots.v, dots.down, dots.v;
    x_(N,1), x_(N,2), dots.h, x_(N,D);
  ) = mat(mb(x)_1^T; mb(x)_2^T; dots.v; mb(x)_N^T) in underbrace(RR^(N times D), "N Tokens" times "D Features")
$

Gli elementi *$x_(n i)$* dei token (celle della matrice) sono chiamati *feature*. Il blocco fondamentale di un Transformer è una funzione:
$
  tilde(X) = "TransformerLayer"(X)
$
con $tilde(mb(X)) in RR^(N times D)$ (stessa dimensionalità dell'input). L'idea è quella di andare a concatenare *più layer* di questo tipo, ognuno con *parametri indipendenti*, per costruire una rete profonda che trasforma progressivamente la rappresentazione dei token.

I parametri di ogni layer possono essere appresi tramite backpropagation, ottimizzando una loss task-specific (es. cross-entropy per classificazione o generazione).

=== Calcolo delle similarità

L'attenzione inizia calcolando le *similarità dot-product* tra tutti i token. Per ogni coppia di token *$m$* e *$n$*, si calcola:
$
  s_(m n) = mb(x)_m^T mb(x)_n
$
il risultato $s_(m n)$ è un punteggio che indica quanto i due token sono *_affini_* o _simili_ tra loro. Un punteggio più alto indica una maggiore similarità.

In forma matriciale (*matrice di similarità*) viene creata una matrice quadrata $N times N$, in modo da considerare tutte le coppie di token:
$
  mb(S) = mb(X) mb(X)^T in RR^(N times N)
$

=== Normalizzazione Softmax

Le similarità grezze vengono convertite in *pesi di attenzione* tramite softmax (applicata per righe):
$
  a_(m n) = e^(s_(m n)) / (sum_j e^(s_(m j)))
$

In forma matriciale:
$
  mb(A) = "Softmax"(mb(X) mb(X)^T)
$

#nota()[
  La softmax normalizza i punteggi di similarità, garantendo due proprietà chiave:
  - *Sommatoria a 1*: Tutte le righe di $mb(A)$ sommano a 1, quindi i *pesi di attenzione* rappresentano una *distribuzione di probabilità sui token*:
  $
    sum_(m=1)^N a_(m n) = 1, quad forall m in {1, dots, N}
  $

  - *Non negatività*: Tutti gli elementi $a_(m n) >= 0$. Questo significa che *ogni token contribuisce* positivamente *alla rappresentazione finale*, con un peso che riflette la sua rilevanza.
]

=== Output del layer Transformer

L'output è una combinazione pesata di tutti i vettori value:
$
  y_m = sum_(n=1)^N underbrace(mr(a_(m n)),"peso "\ "attenzione") underbrace(mb(w_n), "embedding" \ "token" n)
$

In forma matriciale:
$
  Y = mr(A) mb(X) = "Softmax"[mb(X) mb(X)^T] mb(X)
$

#informalmente()[
  Ogni $mb(y)_m$ è una *media pesata* di tutti gli $mb(x)_n$. L'attenzione apprende quali token sono rilevanti per ciascun output.
]

=== Key, Query e Value

Siccome il calcolo precedente $S = X X^T$ considera solamente la similarità tra embedding (*similarità vettoriale*), non permette al modello di *apprendere* cosa significa similarità o considerare la similarità semantica.

Per questo motivo, nei Transformer si introducono *proiezioni lineari apprese* per query, key e value. Ogni token $x_n$ viene proiettato in tre spazi distinti:
- *Query* (cosa cerco)
- *Key* (cosa è disponibile)
- *Value* (l'informazione che recupero)


#informalmente()[
  *Analogia con il recupero di informazioni*: immagina di cercare un film online. Ogni film ha attributi descrittivi (vettore *key*) e un contenuto (vettore *value*). L'utente esprime le sue preferenze (vettore *query*). Il sistema confronta la query con tutte le key, trova la corrispondenza più vicina e restituisce il value.
]

Nei Transformer si introducono *proiezioni apprendibili*:
$
  mb(Q) = mb(X) mb(W)^((q)), quad mb(W)^((q)) in RR^(D times D_k) \
  mb(K) = mb(X) mb(W)^((k)), quad mb(W)^((k)) in RR^(D times D_k) \
  mb(V) = mb(X) mb(W)^((v)), quad mb(W)^((v)) in RR^(D times D_v)
$

Dove *$D_k$* e *$D_v$* sono le dimensioni delle query/key e value rispettivamente. Solitamente hanno una dimensione più piccola di $D$ per ridurre il costo computazionale. 

#attenzione()[
  $D_k$ e $D_q$ devono corrispondere per permettere il calcolo del prodotto scalare. $D_v$ può essere diverso, poiché rappresenta l'informazione che viene recuperata, non la similarità. Solitamente $D_v$ è molto simile a $D_k + D_v$
]

La *matrice di similarità* diventa:
$
  S = mr(Q) mr(K)^T
$
Viene computato un dot-product tra ogni possibile di coppia di query e key, producendo una matrice $N times N$ di punteggi di similarità.

Sucessivamente viene calcolata una somma pesata colonna per colonna. *L'output* finale è dato da:
$
  A = "Softmax"(mr(Q) mr(K)^T), quad mb(Y) = A mr(V)
$
Dove $A$ avrà una dimensione $N times N$ e $mb(Y)$ avrà dimensione $N times D_v$ (scegliendo $D_v = D$ otteniamo $mb(Y) in R^(N times D)$).

Graficamente le operazioni possono essere rappresentate come segue:
#align(center)[
  #cetz.canvas({
    import cetz.draw: *
    
    let box-width = 2.5
    let box-height = 0.6
    
    // Input X
    rect((0, 0), (box-width, box-height), stroke: black, fill: white, name: "X")
    content((box-width/2, box-height/2), text(size: 11pt, weight: "bold", $mb(X) in R^(N times D)$))
    
    // Matrici di peso W^(q), W^(k), W^(v)
    let y1 = 1.8
    rect((-1.5, y1), (-0.3, y1 + box-height), stroke: blue.darken(20%), 
         fill: blue.lighten(70%), name: "Wq")
    content((-0.9, y1 + box-height/2), text(size: 10pt, $mb(W)^((q))$))
    
    rect((0.4, y1), (1.6, y1 + box-height), stroke: blue.darken(20%), 
         fill: blue.lighten(70%), name: "Wk")
    content((1.0, y1 + box-height/2), text(size: 10pt, $mb(W)^((k))$))
    
    rect((2.3, y1), (3.5, y1 + box-height), stroke: blue.darken(20%), 
         fill: blue.lighten(70%), name: "Wv")
    content((2.9, y1 + box-height/2), text(size: 10pt, $mb(W)^((v))$))
    
    // Frecce da X a W
    line((box-width/2, box-height), (-0.9, y1), mark: (end: "stealth"))
    content((-1.2, 1.2), text(size: 9pt, $mb(Q) in R^(D times D_q)$))
    
    line((box-width/2, box-height), (1.0, y1), mark: (end: "stealth"))
    content((1.3, 1.3), text(size: 9pt, $mb(K) in R^(D times D_k)$))
    
    line((box-width/2, box-height), (2.9, y1), mark: (end: "stealth"))
    content((3.3, 1.2), text(size: 9pt, $mb(V) in R^(D times D_v)$))

    content((1.6, 3.4), text(size: 9pt, $in R^(N times N)$))

    content((4.5, 2.1), text(size: 9pt, $W^v in R^(N times D_v)$))
    
    // MatMul (Q * K^T)
    let y2 = 3.2
    rect((-0.9, y2), (0.9, y2 + box-height), stroke: black, 
         fill: rgb("#f4c9a6"), radius: 0.15cm)
    content((0, y2 + box-height/2), text(size: 10pt, [mat mul]))
    
    line((-0.9, y1 + box-height), (-0.5, y2), mark: (end: "stealth"))
    line((1.0, y1 + box-height), (0.5, y2), mark: (end: "stealth"))
    
    // Scale
    let y3 = 4.4
    rect((-0.9, y3), (0.9, y3 + box-height), stroke: black, 
         fill: rgb("#fff8dc"), radius: 0.15cm)
    content((0, y3 + box-height/2), text(size: 10pt, [scale]))
    
    line((0, y2 + box-height), (0, y3), mark: (end: "stealth"))
    
    // Softmax
    let y4 = 5.6
    rect((-0.9, y4), (0.9, y4 + box-height), stroke: black, 
         fill: rgb("#b8e6b8"), radius: 0.15cm)
    content((0, y4 + box-height/2), text(size: 10pt, [softmax]))
    
    line((0, y3 + box-height), (0, y4), mark: (end: "stealth"))
    
    // MatMul finale (A * V)
    let y5 = 6.8
    rect((-0.9, y5), (0.9, y5 + box-height), stroke: black, 
         fill: rgb("#f4c9a6"), radius: 0.15cm)
    content((0, y5 + box-height/2), text(size: 10pt, [mat mul]))
    
    line((0, y4 + box-height), (0, y5), mark: (end: "stealth"))
    
    // Freccia curva da V a mat mul finale
    line((2.9, y1 + box-height), (2.9, y5 + box-height/2), stroke: (dash: "dashed"))
    line((2.9, y5 + box-height/2), (0.9, y5 + box-height/2), mark: (end: "stealth"))
    
    // Output Y
    let y6 = 8.2
    rect((-0.5, y6), (0.5, y6 + box-height), stroke: black, fill: white)
    content((0, y6 + box-height/2), text(size: 11pt, weight: "bold", $mb(Y)$))
    
    line((0, y5 + box-height), (0, y6), mark: (end: "stealth"))
  })
]

#nota[
  La matrice _value_ $V = W^v$, viene moltiplicata per la matrice di attenzione $A$ per produrre l'output finale $Y$, ovvero l'embedding arriccheto del contesto. Ogni riga di $Y$ è una combinazione pesata delle righe di $V$, con i pesi dati da $A$ (somma pesata vettoriale). 

  Per ogni token $i$ (una riga della matrice output $Y$), il nuovo embedding $y_i$ è costruito prendendo un po' del valore di ogni token $j$ della sequenza, in base a quanto $i$ ha prestato attenzione a $j$:
  $ 
    mb(y)i = a(i,1) v_1 + a(i,2) v_2 + dots + a(i,N) v_N
  $
]



=== Scaled Dot-Product Self-Attention



$mr("Problema")$: Se gli elementi dei vettori query e key hanno media $0$ e varianza $1$, il loro prodotto scalare ha varianza $D_k$ (le varianze si sommano). Una varianza troppo alta porta i numeri in un intervallo molto ampio. 

Un'alta varianza porta a input grandi per la softmax, che diventa molto piatta e produce *gradienti molto piccoli* (funzione esponenziale).

La $mg("soluzione")$ è scalare i punteggi di similarità dividendo per $sqrt(D_k)$, che mantiene i valori in un intervallo più gestibile:
  $
    "Attention"(Q, K, V) = "Softmax"((Q K^T) / sqrt(D_k)) V
  $

Questa operazione permette di *stabilizza la softmax*, rendendo l'attenzione robusta all'aumentare della dimensionalità.

#align(center)[
  #cetz.canvas(length: 1cm, {
    import cetz.draw: *
    
    // Funzione gaussiana
    let gaussian(x, sigma) = {
      calc.pow(calc.e, -calc.pow(x, 2) / (2 * calc.pow(sigma, 2))) / (sigma * calc.sqrt(2 * calc.pi))
    }
    
    // Funzione per disegnare una curva gaussiana
    let draw-gaussian(origin, sigma, scale, color, fill-tails: false) = {
      let points = ()
      let x-min = -3.0
      let x-max = 3.0
      let steps = 100
      let dx = (x-max - x-min) / steps
      
      for i in range(steps + 1) {
        let x = x-min + i * dx
        let y = gaussian(x, sigma) * scale
        points.push((origin.at(0) + x, origin.at(1) + y))
      }
      
      // Disegna la curva
      line(..points, stroke: (paint: color, thickness: 1.5pt))
      
      // Riempi le code se richiesto
      if fill-tails {
        // Coda sinistra
        let left-points = ((origin.at(0) - 3, origin.at(1)),)
        for i in range(steps + 1) {
          let x = x-min + i * dx
          if x < -1.5 {
            let y = gaussian(x, sigma) * scale
            left-points.push((origin.at(0) + x, origin.at(1) + y))
          }
        }
        left-points.push((origin.at(0) - 1.5, origin.at(1)))
        line(..left-points, stroke: none, fill: red.lighten(70%), close: true)
        
        // Coda destra
        let right-points = ((origin.at(0) + 1.5, origin.at(1)),)
        for i in range(steps + 1) {
          let x = x-min + i * dx
          if x >= 1.5 {
            let y = gaussian(x, sigma) * scale
            right-points.push((origin.at(0) + x, origin.at(1) + y))
          }
        }
        right-points.push((origin.at(0) + 3, origin.at(1)))
        line(..right-points, stroke: none, fill: red.lighten(70%), close: true)
      }
    }
    
    // --- Grafico sinistra: Without Scaling ---
    set-origin((-8, 0))
    
    // Titolo
    content((0, 3.8), text(size: 10pt, weight: "bold", [Senza Scaling]))
    
    // Assi
    line((-3.2, 0), (3.2, 0), stroke: (paint: gray, thickness: 0.8pt))
    line((0, 0), (0, 3), stroke: (paint: gray, thickness: 0.8pt), mark: (end: ">"))
    
    // Tick marks
    for x in (-2, -1, 0, 1, 2) {
      line((x, -0.1), (x, 0.1), stroke: (thickness: 0.8pt))
      content((x, -0.4), text(size: 8pt, str(x)))
    }
    
    // Gaussiana larga con code evidenziate
    draw-gaussian((0, 0), 1.5, 6, black, fill-tails: true)
    
    // Label sotto
    content((0, -1.5), text(size: 8pt)[
      Variance = $D_k$. Large dot products push\ 
      softmax into regions with #text(fill: red)[vanishing gradients].
    ])
    
    // Freccia verso l'alto
    line((0, 3.5), (0, 3.1), mark: (start: "stealth"), stroke: (thickness: 1.2pt))
    
    // --- Box centrale: Formula ---
    set-origin((4, 0))
    
    rect((-1.8, 1.2), (1.8, 2.2), stroke: (paint: blue.darken(20%), thickness: 1.2pt), 
         fill: blue.lighten(90%), radius: 0.1cm)
    content((0, 1.7), text(size: 10pt, $"Score" = (mb(Q) mb(K)^T) / sqrt(D_k)$))
    
    // --- Grafico destra: With Scaling ---
    set-origin((4, 0))
    
    // Titolo
    content((0, 3.8), text(size: 10pt, weight: "bold", [ Con Scaling]))
    
    // Freccia verso l'alto
    line((0, 3.5), (0, 3.1), mark: (start: "stealth"), stroke: (thickness: 1.2pt))
    
    // Assi
    line((-3.2, 0), (3.2, 0), stroke: (paint: gray, thickness: 0.8pt))
    line((0, 0), (0, 3), stroke: (paint: gray, thickness: 0.8pt), mark: (end: ">"))
    
    // Tick marks
    for x in (-2, -1, 0, 1, 2) {
      line((x, -0.1), (x, 0.1), stroke: (thickness: 0.8pt))
      content((x, -0.4), text(size: 8pt, str(x)))
    }
    
    // Gaussiana stretta (più alta e concentrata)
    draw-gaussian((0, 0), 0.7, 6, black)
    
    // Label sotto
    content((0, -1.5), text(size: 8pt)[
      Variance = 1. Dividing by $sqrt(D_k)$\
      stabilizes the training dynamics.
    ])
  })
]

== Multi-Head Attention

Una singola testa di attenzione può modellare solo *un tipo di relazione* alla volta. Ma il linguaggio naturale (e altri dati) contiene molteplici pattern simultanei:
- struttura grammaticale
- ruoli semantici
- tempo verbale
- dipendenze a lunga distanza
- relazioni lessicali

*Idea*: usare più teste di attenzione in parallelo, ognuna con le proprie proiezioni apprese:
$
  mb(Q)_h = mb(X) mb(W)_h^((q)), quad
  mb(K)_h = mb(X) mb(W)_h^((k)), quad
  mb(V)_h = mb(X) mb(W)_h^((v))
$

Ogni testa calcola la scaled dot-product attention:
$
  mb(H)_h = "Attention"(mb(Q)_h, mb(K)_h, mb(V)_h)
$

Le teste vengono poi *concatenate* e proiettate con una trasformazione lineare finale:
$
  mb(Y)(mb(X)) = "Concat"(mb(H)_1, dots, mb(H)_H) mb(W)^((o))
$

#nota()[
  *Dimensioni*: ogni testa produce $mb(H)_h in RR^(N times D_v)$. Con $H$ teste:
  $
    "Concat"(mb(H)_1, dots, mb(H)_H) in RR^(N times (H D_v))
  $
  La proiezione finale $mb(W)^((o)) in RR^(H D_v times D)$ riporta al embedding dimension del modello.

  Scegliendo $D_v = D / H$, la matrice concatenata ha dimensione $(N times D)$, uguale all'input. Tutte le matrici $mb(W)_h^((q)), mb(W)_h^((k)), mb(W)_h^((v)}, mb(W)^((o))$ vengono apprese congiuntamente.
]

#informalmente()[
  Le diverse teste imparano *modi diversi di attendere*. Il modello cattura così più tipi di struttura contemporaneamente, in modo analogo ai filtri multipli in un layer CNN.
]

== Self-Attention e Cross-Attention

Un layer di attenzione standard prende in input due sequenze $mb(X)$ e $mb(X)'$ e calcola:
$
  mb(Q) = mb(X) mb(W)^((q)), quad
  mb(K) = mb(X)' mb(W)^((k)), quad
  mb(V) = mb(X)' mb(W)^((v))
$
$
  mb(A) = "Softmax"(mb(Q) mb(K)^T / sqrt(D_k)), quad
  mb(Y) = mb(A) mb(V)
$

/ Self-attention: $mb(X) = mb(X)'$ — la stessa sequenza è usata per query, key e value.
/ Cross-attention: $mb(X) != mb(X)'$ — query dalla sequenza di output, key e value dalla sequenza di input (encoder).

#nota()[
  La *cross-attention* esiste solo nei Transformer encoder–decoder, come:
  - Modelli di machine translation
  - Modelli di summarization
  - Modelli sequence-to-sequence (T5, BART)
]

== Transformer Layer: Residui, LayerNorm e MLP

Per migliorare la stabilità e l'efficienza del training, il blocco di multi-head attention è arricchito con:

/ Connessione residua: garantisce che l'output mantenga la stessa forma dell'input $mb(X) in RR^(N times D)$.

/ Layer normalization: applicata dopo il blocco di attenzione (*post-norm*):
  $
    mb(Z) = "LayerNorm"[mb(Y)(mb(X)) + mb(X)]
  $
  oppure prima (*pre-norm*):
  $
    mb(Z) = mb(Y)(mb(X)_0) + mb(X), quad mb(X)_0 = "LayerNorm"(mb(X))
  $

/ MLP position-wise: l'output dell'attenzione è una combinazione lineare di vettori input. Per aumentare l'espressività, ogni token viene passato attraverso lo stesso MLP non-lineare:
  - Post-norm: $tilde(mb(X)) = "LayerNorm"["MLP"(mb(Z)) + mb(Z)]$
  - Pre-norm: $tilde(mb(X)) = "MLP"(mb(Z)_0) + mb(Z), quad mb(Z)_0 = "LayerNorm"(mb(Z))$

#informalmente()[
  Un layer Transformer completo combina:

  *Multi-Head Attention → Residuo → Norm → MLP → Residuo → Norm*

  Più layer di questo tipo vengono impilati con parametri indipendenti per costruire reti profonde.
]

== Positional Encoding

#attenzione()[
  La self-attention tratta tutti i token di input in modo *identico* rispetto alla loro posizione: riordinare l'input riordina semplicemente l'output. Questo è un vantaggio per il parallelismo, ma un *problema per le sequenze* dove l'ordine porta significato.

  Esempio: "Il cibo era cattivo, non buono." e "Il cibo era buono, non cattivo." hanno gli stessi token ma significato opposto.
]

=== Soluzione: aggiungere vettori posizionali

Per ogni posizione $n$, si crea un vettore $mb(r)_n$ e si forma:
$
  mb(x)'_n = mb(x)_n + mb(r)_n
$

*Requisiti per un buon positional encoding*:
- Unico per ogni posizione
- Limitato (bounded)
- Generalizza a sequenze più lunghe di quelle viste durante il training
- Rappresenta la *distanza relativa*, non solo l'indice assoluto

Le *funzioni sinusoidali a frequenze multiple* soddisfano tutte queste proprietà. Per un token alla posizione $n$, il vettore di codifica posizionale $mb(r)_n$ ha componenti:
$
  r_(n,i) = cases(
    sin(L^(i \/ D) n) & "se" i "è pari",
    cos(L^((i-1) \/ D) n) & "se" i "è dispari"
  )
$

#nota()[
  *Proprietà chiave*:
  - Ogni dimensione codifica la posizione con una *lunghezza d'onda diversa*.
  - Le dimensioni inferiori variano lentamente → catturano struttura a lungo raggio.
  - Le dimensioni superiori variano rapidamente → catturano struttura fine.
  - La combinazione codifica *univocamente* la posizione rimanendo bounded.
  - Le posizioni relative possono essere inferite dalle *differenze di fase* dei sinusoidi.

  Mescolando sin e cos a frequenze scalate esponenzialmente, il modello ottiene una rappresentazione continua e liscia dell'ordine dei token che si generalizza oltre la lunghezza di training.
]

== Reti Transformer

Vaswani et al. (2017) proposero un modello senza operazioni convoluzionali né ricorrenti, composto *esclusivamente da layer di attenzione*. Il Transformer completo è composto da:

- Un *encoder* che combina $N=6$ moduli, ognuno con un sottomodulo di multi-head attention e un MLP a un layer nascosto per token, con connessioni residue e layer normalization.
- Un *decoder* con struttura simile, ma con layer di attenzione *causale* (masked) e layer di cross-attention che attendono alle key e value finali dell'encoder.

=== Transformer Encoder

L'encoder processa una sequenza di input e produce una rappresentazione contestuale più ricca.

*Input*: $mb(X) in RR^(N times D)$, con $N$ = numero di token e $D$ = dimensione dell'embedding.

Ogni layer dell'encoder è composto da:
+ Multi-head self-attention
+ Connessione residua + LayerNorm
+ MLP position-wise
+ Connessione residua + LayerNorm

Ogni layer preserva la dimensionalità: input $N times D arrow$ output $N times D$.

// Nota: il diagramma testuale viene descritto narrativamente
// poiché richiederebbe asset grafici specifici del corso

#informalmente()[
  L'encoder costruisce *rappresentazioni contestuali dei token*:
  - Ogni token attende a tutti gli altri token.
  - Le dipendenze a lungo raggio vengono catturate direttamente.
  - Il significato delle parole diventa dipendente dal contesto.

  Esempio: in "L'animale era stanco perché aveva corso", il token "aveva" attende fortemente ad "animale".

  L'impilamento dei layer raffina progressivamente le rappresentazioni. L'output finale dell'encoder ha la stessa forma dell'input ma con *struttura semantica molto più ricca*.
]

==== Pre-training dell'encoder: Masked Language Modeling

Un sottoinsieme casuale di token (es. il 15%) viene sostituito con un token speciale `[MASK]`. Il modello viene addestrato a *predire i token mancanti* dato il contesto:

#esempio()[
  `Sono andato [MASK] il fiume per raggiungere la banca.`

  Il modello deve predire la parola mascherata dal contesto circostante. Una volta addestrato, l'encoder può essere *fine-tuned* per task di classificazione usando il primo token di output (corrispondente a `[CLS]`).
]

=== Transformer Decoder

Il decoder genera sequenze di output *un token alla volta*. Un decoder-only Transformer viene usato come *modello generativo*.

*Input*:
- Token generati in precedenza (shifted right)
- Rappresentazioni dell'encoder (per modelli encoder–decoder)

Ogni layer del decoder contiene:
+ Masked multi-head self-attention
+ Connessione residua + LayerNorm
+ Cross-attention (encoder–decoder attention)
+ Connessione residua + LayerNorm
+ MLP position-wise
+ Connessione residua + LayerNorm

/ Masked self-attention: garantisce che un token possa attendere *solo ai token precedenti* (maschera causale). Nessun accesso alle informazioni future.

#esempio()[
  *GPT (Generative Pretrained Transformer)*: il decoder apprende le probabilità condizionali:
  $
    p(x_n | x_1, dots, x_(n-1))
  $
  - Input: i primi $n-1$ token
  - Output: distribuzione di probabilità sul token $x_n$
  - Si campiona $x_n$, si appende alla sequenza, si ripete.
]

==== Cosa apprende il decoder

Il decoder esegue *generazione autoregressiva*. Per ogni posizione $t$:
- Attende ai token generati in precedenza.
- Opzionalmente attende agli output dell'encoder (in task sequence-to-sequence).
- Predice la distribuzione di probabilità del token successivo.
- L'ultimo layer produce logit → softmax → probabilità del prossimo token.

Questo abilita: generazione di testo, traduzione, sistemi di dialogo, generazione di codice.

=== Encoder vs Decoder: confronto

#table(
  columns: (auto, 1fr, 1fr),
  [*Caratteristica*], [*Encoder*], [*Decoder*],
  [Ruolo principale], [Costruire rappresentazioni contestuali], [Generare sequenza di output],
  [Tipo di attenzione], [Full self-attention], [Masked self-attention],
  [Accesso a token futuri], [Sì], [No (maschera causale)],
  [Cross-attention], [No], [Sì (attende l'output dell'encoder)],
  [Usato in], [BERT, Vision Transformer], [GPT, modelli di traduzione],
  [Output], [Feature interne ricche], [Predizione del prossimo token],
)

#informalmente()[
  - *Encoder* = Capire
  - *Decoder* = Generare
  - *Transformer completo* = Encoder → Decoder
]

== Esempio: Machine Translation

#esempio()[
  - Input (inglese): _"The cat is sleeping."_
  - Output (olandese): _"De kat slaapt."_

  *Step 1 — Encoding*: l'encoder mappa l'intera frase di input in una rappresentazione interna:
  $
    mb(Z) = "Encoder"(x_1, dots, x_N)
  $

  *Step 2 — Decoding*: il decoder genera i token di output uno alla volta, condizionato su:
  - I token di output generati in precedenza
  - La rappresentazione codificata $mb(Z)$

  Questo condizionamento è realizzato tramite *cross-attention*, dove:
  - Query ($mb(Q)$): stato corrente del decoder (cosa sto cercando)
  - Key ($mb(K)$): output dell'encoder (cosa è disponibile)
  - Value ($mb(V)$): informazione che recupero dall'encoder

  *Training*: usa coppie sequenza input–output, ottimizzate con cross-entropy loss sui token generati.
]

#nota()[
  Questa architettura è usata in:
  - Traduzione automatica
  - Summarization
  - Speech-to-text
  - Task sequence-to-sequence generali
]

== Training: Synthetic Copy Dataset

Per verificare il corretto funzionamento di un Transformer end-to-end, si usa spesso un task sintetico: il modello deve *copiare una sequenza*.

=== Token speciali

/ `PAD = 0`: padding
/ `BOS = 1`: beginning of sequence
/ `EOS = 2`: end of sequence
/ Token payload: $in {3, dots, K-1}$

=== Struttura di un campione

Sia $S$ la lunghezza della sequenza sorgente. Un campione contiene:

*Encoder input (source)*:
$
  "src" = [x_1, x_2, dots, x_S]
$

*Decoder input (shifted right)*:
$
  "decoder\_input" = ["BOS", x_1, x_2, dots, x_S]
$

*Label (next-token targets)*:
$
  "label" = [x_1, x_2, dots, x_S, "EOS"]
$

#informalmente()[
  Questo task sintetico è utile perché:
  - Ha un obiettivo semplice, facile da debuggare.
  - Conferma che masking, attenzione e loss siano cablati correttamente.
  - La loss dovrebbe scendere rapidamente se l'architettura è corretta.
]

=== Training loop (teacher forcing)

Per ogni batch:
+ *Costruire le maschere* (padding + causale per il decoder): `src_mask`, `tgt_mask`
+ *Encoder* costruisce la memoria: $mb(E) = "encode"("src", "src\_mask")$
+ *Decoder* predice tutti i time step in parallelo: $mb(D) = "decode"(mb(E), "src\_mask", "decoder\_input", "tgt\_mask")$
+ *Proiezione* al vocabolario: $"logits" in RR^(B times T times K)$
+ *Ottimizzazione* con cross-entropy su tutte le posizioni:
  $
    cal(L) = sum_(b,t) "CE"("logits"_(b,t), "label"_(b,t))
  $

=== Greedy Decoding (inference)

+ Inizia con $y_1 = "BOS"$
+ Poi ripete:
  - predice il prossimo token con $arg max$ sui logit
  - appende il token alla sequenza
  - si ferma quando viene generato `EOS` (o si raggiunge la lunghezza massima)

Comportamento atteso: dopo il training, il modello genera
$
  ["BOS", x_1, dots, x_S, "EOS"]
$
ovvero, apprende a copiare la sequenza di input.

#nota()[
  Il *teacher forcing* durante il training significa che il decoder riceve i token di ground truth come input (anche se ha predetto il token sbagliato al passo precedente). Questo accelera la convergenza ma può causare una discrepanza tra training e inference (*exposure bias*).
]