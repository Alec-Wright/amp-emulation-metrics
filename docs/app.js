// ---------------------------------------------------------------------
// Fitted piecewise-linear model parameters
// ---------------------------------------------------------------------
const MODEL = {
  m: -69.4,
  d0: 0.464
};

function predictedMushraFromLogMfcc(x) {
  const y = x <= MODEL.d0 ? 100 : 100 + MODEL.m * (x - MODEL.d0);
  return Math.max(0, Math.min(100, y));
}

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return Number(value).toFixed(digits);
}

// ---------------------------------------------------------------------
// Plot state
// ---------------------------------------------------------------------
let plotInitialized = false;

// Trace order:
//   0 = fitted line
//   1 = demo examples
//   2 = selected example highlight
const SELECTED_TRACE_INDEX = 2;

// ---------------------------------------------------------------------
// DOM elements
// ---------------------------------------------------------------------
const refAudio = document.getElementById("reference-audio");
const emuAudio = document.getElementById("emulation-audio");
const titleEl = document.getElementById("example-title");
const metadataEl = document.getElementById("metadata");
const loopToggle = document.getElementById("loop-toggle");

document.getElementById("play-reference").addEventListener("click", () => {
  playFromStart(refAudio);
});

document.getElementById("play-emulation").addEventListener("click", () => {
  playFromStart(emuAudio);
});

loopToggle.addEventListener("change", () => {
  refAudio.loop = loopToggle.checked;
  emuAudio.loop = loopToggle.checked;
});

function playFromStart(audioEl) {
  refAudio.pause();
  emuAudio.pause();

  refAudio.currentTime = 0;
  emuAudio.currentTime = 0;

  audioEl.play();
}

function highlightExample(example) {
  if (!plotInitialized) {
    return;
  }

  Plotly.restyle(
    "plot",
    {
      x: [[example.log_mfcc_l1]],
      y: [[example.predicted_mushra]],
      text: [[example.label ?? example.id]],
      customdata: [[example.index]]
    },
    [SELECTED_TRACE_INDEX]
  );
}

function selectExample(example) {
  titleEl.textContent = example.label ?? example.id;

  refAudio.src = example.reference_audio;
  emuAudio.src = example.emulation_audio;

  metadataEl.innerHTML = `
    <dl>
      <dt>Example</dt>
      <dd>${example.id}</dd>

      <dt>Model</dt>
      <dd>${example.Model ?? "—"}</dd>

      <dt>Folder</dt>
      <dd>${example["Folder Name"] ?? "—"}</dd>

      <dt>Clip ID</dt>
      <dd>${example.clip_id ?? "—"}</dd>

      <dt>MFCC L1</dt>
      <dd>${fmt(example.mfcc_l1, 4)}</dd>

      <dt>log<sub>10</sub> MFCC L1</dt>
      <dd>${fmt(example.log_mfcc_l1, 3)}</dd>

      <dt>Predicted MUSHRA</dt>
      <dd>${fmt(example.predicted_mushra, 1)}</dd>
    </dl>
  `;

  highlightExample(example);
}

// ---------------------------------------------------------------------
// Load examples and draw plot
// ---------------------------------------------------------------------
async function main() {
  const examples = await fetch(`./data/examples.json?v=${Date.now()}`).then(response => {
    if (!response.ok) {
      throw new Error(`Could not load examples.json: ${response.status}`);
    }
    return response.json();
  });

  const processed = examples.map((d, i) => {
    const logMfcc =
      d.log_mfcc_l1 !== undefined && d.log_mfcc_l1 !== null
        ? Number(d.log_mfcc_l1)
        : Math.log10(Number(d.mfcc_l1));

    const predicted =
      d.predicted_mushra !== undefined && d.predicted_mushra !== null
        ? Number(d.predicted_mushra)
        : predictedMushraFromLogMfcc(logMfcc);

    return {
      ...d,
      index: i,
      log_mfcc_l1: logMfcc,
      predicted_mushra: predicted
    };
  });

  if (processed.length === 0) {
    throw new Error("examples.json contains no examples.");
  }

  const xs = processed.map(d => d.log_mfcc_l1);
  const minX = Math.min(...xs, MODEL.d0) - 0.1;
  const maxX = Math.max(...xs, MODEL.d0) + 0.1;

  const fitX = [];
  const fitY = [];

  for (let i = 0; i <= 300; i++) {
    const x = minX + (i / 300) * (maxX - minX);
    fitX.push(x);
    fitY.push(predictedMushraFromLogMfcc(x));
  }

  const fitTrace = {
    x: fitX,
    y: fitY,
    type: "scatter",
    mode: "lines",
    name: "Piecewise-linear fit",
    line: {
      color: "#222",
      width: 3
    },
    hovertemplate:
      "log10 MFCC L1: %{x:.3f}<br>" +
      "Predicted MUSHRA: %{y:.1f}<extra></extra>"
  };

  const pointTrace = {
    x: processed.map(d => d.log_mfcc_l1),
    y: processed.map(d => d.predicted_mushra),
    type: "scatter",
    mode: "markers",
    name: "Demo examples",
    customdata: processed.map(d => d.index),
    text: processed.map(d => d.label ?? d.id),
    marker: {
      size: 13,
      color: processed.map(d => d.predicted_mushra),
      cmin: 0,
      cmax: 100,
      colorscale: "Viridis",
      line: {
        color: "white",
        width: 1.5
      },
      colorbar: {
        title: "Predicted<br>MUSHRA"
      }
    },
    hovertemplate:
      "<b>%{text}</b><br>" +
      "log10 MFCC L1: %{x:.3f}<br>" +
      "Predicted MUSHRA: %{y:.1f}<br>" +
      "Click to listen<extra></extra>"
  };

  const selectedTrace = {
    x: [],
    y: [],
    type: "scatter",
    mode: "markers",
    name: "Selected example",
    customdata: [],
    text: [],
    marker: {
      size: 22,
      symbol: "circle-open",
      color: "#d62728",
      line: {
        color: "#d62728",
        width: 4
      }
    },
    hovertemplate:
      "<b>Selected: %{text}</b><br>" +
      "log10 MFCC L1: %{x:.3f}<br>" +
      "Predicted MUSHRA: %{y:.1f}<extra></extra>",
    showlegend: true
  };

  const layout = {
    template: "plotly_white",
    margin: {
      l: 70,
      r: 30,
      t: 30,
      b: 70
    },
    xaxis: {
      title: "log10 MFCC L1 distance",
      zeroline: false
    },
    yaxis: {
      title: "Predicted MUSHRA-style similarity score",
      range: [-2, 103],
      dtick: 10
    },
    legend: {
      orientation: "h",
      y: -0.2
    },
    shapes: [
      {
        type: "line",
        x0: MODEL.d0,
        x1: MODEL.d0,
        y0: 0,
        y1: 100,
        line: {
          color: "#999",
          width: 1.5,
          dash: "dot"
        }
      },
      {
        type: "rect",
        xref: "paper",
        x0: 0,
        x1: 1,
        y0: 90,
        y1: 100,
        fillcolor: "rgba(68, 170, 153, 0.08)",
        line: {
          width: 0
        },
        layer: "below"
      }
    ],
    annotations: [
      {
        x: MODEL.d0,
        y: 5,
        text: "breakpoint",
        showarrow: false,
        xanchor: "left",
        yanchor: "bottom",
        font: {
          size: 12,
          color: "#666"
        }
      }
    ]
  };

  const config = {
    responsive: true,
    displaylogo: false
  };

  await Plotly.newPlot(
    "plot",
    [fitTrace, pointTrace, selectedTrace],
    layout,
    config
  );

  plotInitialized = true;

  const plotEl = document.getElementById("plot");

  plotEl.on("plotly_click", event => {
    const point = event.points[0];

    if (point.customdata !== undefined && point.customdata !== null) {
      selectExample(processed[point.customdata]);
      return;
    }

    // If the fitted line is clicked instead of a marker, select the nearest example.
    const clickedX = point.x;

    const nearest = processed.reduce((best, current) => {
      const bestDist = Math.abs(best.log_mfcc_l1 - clickedX);
      const currentDist = Math.abs(current.log_mfcc_l1 - clickedX);
      return currentDist < bestDist ? current : best;
    }, processed[0]);

    selectExample(nearest);
  });

  // Load and highlight the first example by default.
  selectExample(processed[0]);
}

main().catch(error => {
  console.error(error);

  document.getElementById("plot").innerHTML = `
    <p class="error">
      Could not load the demo. Check that <code>data/examples.json</code> exists
      and that all audio paths are correct.
    </p>
  `;
});

