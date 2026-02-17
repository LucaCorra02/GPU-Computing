#import "../template.typ": *

= Stream e gestione della memoria

== Trasferimento di memoria

Quando la GPU richiede un trasferimento di dati dall'host i dati si devono trovare necessariamente in una porzione di memoria non paginabile (pinned). Se così non fosse nel caso di uno swap il trasferimento fallirebbe.

Per ovviare a questo problema quando trasferiamo dati (ad esempio con `cudaMemcpy`) viene allocato un buffer di memoria non paginabile dove vengono copiati i dati prima di essere trasferiti.

In alternativa possiamo direttamente allocare sull'host memoria non paginabile, una volta allocata possiamo fare trasferimenti asincroni e aumentando anche la velocità del trasferimento evitando il buffer di pinned memory intermedio (vedremo più avanti come).

=== Concurrency overlap

Uno dei vantaggi è la concurrency overlap. Tramite stream indipendenti abbiamo flussi di esecuzione paralleli: mentre uno stream sta trasferendo dei dati un'altro potrebbe star eseguendo un task. Lo stesso vale per la CPU, possiamo proseguire con l'esecuzione del nostro programma mentre trasferiamo dati.

// TODO: sistemare il disegno

#{
  import fletcher: edge, node
  figure(
    fletcher.diagram(
      node-stroke: 1pt,
      spacing: (0.1cm, 1cm),
      node((0, 0), [*Esecuzione Sincrona*], shape: rect, fill: none, stroke: none),
      node((0.3, 1), [Copy A], fill: orange, stroke: black, width: 1.8cm, height: 0.9cm),
      node((1, 1), [Copy B], fill: orange, stroke: black, width: 1.8cm, height: 0.9cm),
      node((2.7, 1), [Kernel], fill: green, stroke: black, width: 2.5cm, height: 0.9cm),
      node((7.2, 1), [Copy C], fill: orange, stroke: black, width: 1.8cm, height: 0.9cm),
      edge((-0.3, 1), (9.2, 1), "->", stroke: 2pt),
    ),
    caption: [Somma di vettori $C = A + B$ con trasferimenti sequenziali],
  )

  figure(
    fletcher.diagram(
      node-stroke: 1pt,
      spacing: (0.6cm, 0.9cm),
      node((0, 0), [*Esecuzione con Stream*], shape: rect, fill: none, stroke: none),

      // Generazione degli stream tramite loop
      ..for i in range(4) {
        let y = i + 1
        let offset = i * 0.7
        (
          node((-0.5, y), [Stream #(i + 1)], fill: none, stroke: none),
          node((0.8 + offset, y), [A], fill: orange, stroke: black, width: 0.9cm, height: 0.7cm),
          node((1.5 + offset, y), [B], fill: orange, stroke: black, width: 0.9cm, height: 0.7cm),
          node((2.7 + offset, y), [v_s], fill: green, stroke: black, width: 1.3cm, height: 0.7cm),
          node((4.3 + offset, y), [C], fill: orange, stroke: black, width: 0.9cm, height: 0.7cm),
          edge((0, y), (5.2 + offset, y), "->", stroke: 2pt),
        )
      },
    ),
    caption: [Partizionamento dei vettori e sovrapposizione di trasferimento e computazione tramite stream],
  )
}

#attenzione()[
  Il DMA è unico, non posso fare trasferimenti paralleli ma posso far lavorare contemporaneamente CPU e DMA.
  Bisogna anche fare attenzione a limitare le allocazioni di pinned memory, siccome non può essere spostata può influire sul tutto il sistema.
]

=== Gestione pined memory
//AGGIUNGRE MATRICI CON PIN MEMORY

== Stream

Uno stream è un'unità di concorrenza in CUDA. Ogni stream è una coda FIFO di operazioni. Ogni stream è indipendente dagli altri. L'esecuzione di uno stream è sempre asincrona rispetto all'host.

Un pattern comune è utilizzarli per i trasferimenti di memoria. Assegno il trasferimento a uno stream (utilizzando la pinned memory sull'host) e ad un altro assegno un task.

Esistono due stream:
- Default o nulla  $"ID" = 0$
- Stream non nullo $"ID" > 0$

Il default stream si comporta così:
- Se si lancia un kernel nel default stream aspetterà che tutti gli altri stream finiscano.
- Se si lancia un kernel in un altro stream aspetterò che il default finisca.

#attenzione()[
  Possiamo avere due comportamenti diversi per lo stream di default:
  - `--default-stream legacy (or noflag)` lo stream si sincronizza con gli altri stream. È il comportamento descritto in precedenza. Da qui in avanti per il default stream si assume questo comportamento.
  - `--default-stream per-thread` lo stream non si sincronizza con gli altri.
]

Gli stream possono essere usati per aumentare l'occupancy: Ad esempio se il nostro task occupa solo alcuni SM, in vece di sequenzializzare l'esecuzione posso suddividere il problema o eseguire direttamente task diversi su più stream gestiti parallelamente.

```cudaDeviceSynchronize``` sincronizza tutti gli stream. È la sincronizzazione più _forte_ che abbiamo.

//TODO Mettere immagine old vs new

#attenzione()[
  La creazione di uno stream deve avvenire prima del suo utilizzo.
]

È possibile anche sincronizzare il singolo stream con l'host.

Possiamo effettuare delle query sullo stream per sapere se ha completato le operazioni che gli erano state assegnate oppure no.

//Aggiugnere somam di vettori con stream



=== Pinned memory e Stream

La memoria pined viene allocata con `cuda.pinned.array(shape,dtype)`.
```py
arr = np.arrange(1_000_000, dtype=np.float32)
with cuda.pinned(arr):
  d_arr = cuda.to.device(arr)
  #kernel
  out = cuda.device.to.host(d_out)
```

Quando non usare pinned memory:
- Non uso di memoria GPU // TODO: ???
- Non ci sono vantaggi oggettivi // TODO: ???
- Troppi passaggi frequenti tra pinned e unpinned.

Uno stream si crea come `stream=cuda.stream()`. È indipendente dallo stream di default (id non nullo). Per lanciare un kernel sullo stream basta passarlo come parametro.

//aggiungere esempio pipeline loop in cuda
```py
  for i in range(num_streams):
    start = i * chunk_size
    end = start + chunk_size

    #H2D copy
    cuda.to_device(h_in[start:end], to=d_buffers[i], stream=stream[i])

    scale_kernel[blocks, threads, stream[i]](d_buffers[i],210)
```

Essendo le operazioni con gli stream asincrone bisogna ricordarsi di sincronizzare alla fine.

=== Managed memory

Esiste anche un'altro modo per spostare i dati tra GPU e CPU: la managed memory.
È un'astrazione che permette a CPU e GPU di operare in maniera trasparente su una stessa porzione di memoria. In questo caso il driver si occupa di tenere sincronizzati i dati tra CPU e GPU. Può essere una buona soluzione in alcuni casi ma può anche causare overhead significativi e non permette di avere un controllo granulare sui trasferimenti di memoria.

```py
cuda.managed_array(shape,dtype) #da usare con attenzione.
```
