

const fetchWithTimeout = async (resource, options = {}) => {
    const { timeout = 5000 } = options;

    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);

    const response = await fetch(resource, {
        ...options,
        signal: controller.signal
    }).finally(() => clearTimeout(id));

    return response;
};


async function fetchAndPlotData(index, rows, cols) {
    const response = await fetchWithTimeout(`/data/${index}`, { timeout: 3000 });
    const dataBatch = await response.json();

    // for (let i = 0; i < rows; i++) {
    //     for (let j = 0; j < cols; j++) {
    //         Plotly.newPlot(`cell-${i}-${j}`, data['plotting_data'], data['layout'], { responsive: true, autosize: true });
    //     }
    // }

    dataBatch.forEach((data, index) => {
        const rowIndex = Math.floor(index / cols)
        const colIndex = (index % cols)
        Plotly.newPlot(`cell-${rowIndex}-${colIndex}`, data['plotting_data'], data['layout'], { responsive: true, autosize: true });
    });

};


async function initGrid(rows, cols, xlabel, ylabel) {

    // TODO: add containers for xlabel and ylabel and corresponding text
    const figureContainerPrimary = document.getElementById('figure-container-primary');
    const figureContainerSecondary = document.getElementById('figure-container-secondary');
    const subplotContainer = document.getElementById('subplot-container');

    // Set x and y labels
    const xlabelElement = document.getElementById('xlabel');
    xlabelElement.textContent = xlabel;
    const ylabelElement = document.getElementById('ylabel');
    ylabelElement.textContent = ylabel;

    const fragment = document.createDocumentFragment();

    for (let i = 0; i < rows; i++) {
        const row = document.createElement('div');
        row.className = 'row';

        for (let j = 0; j < cols; j++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.id = `cell-${i}-${j}`;
            cell.style.height = `${100 / (rows * 1.2)}vh`;
            cell.style.width = `${100 / cols}vw`;
            row.appendChild(cell);
        }
        fragment.appendChild(row);
    }
    subplotContainer.appendChild(fragment);
}


function initSlider(panels, rows, cols) {

    const slider = document.getElementById('slider');
    slider.max = panels - 1;

    const sliderLabel = document.getElementById('slider-label')

    // function to couple slider changes to plot updates
    // client-side throttling to prevent server overloading
    let lastRequestTime = 0;
    const throttleInterval = 100;

    slider.addEventListener('input', async function(event) {
        const value = event.target.value;
        const now = Date.now();
        if (now - lastRequestTime >= throttleInterval) {
            lastRequestTime = now;
            await fetchAndPlotData(value, rows, cols);
            sliderLabel.textContent = `${value}`
        }
    });
    
}


async function init(rows, cols, panels, xlabel, ylabel) {
    initGrid(rows, cols, xlabel, ylabel);

    // // TODO: figure out why this breaks slider behavior!
    // // Fetch and plot initial data
    await fetchAndPlotData(0, rows, cols);
    initSlider(panels, rows, cols);
};