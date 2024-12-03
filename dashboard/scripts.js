document.addEventListener('DOMContentLoaded', function () {
    const serversList = document.getElementById('servers-list');
    const gpuList = document.getElementById('gpu-list');
    const modelsList = document.getElementById('models-list');
    const gpuChartCtx = document.getElementById('gpu-chart').getContext('2d');
    const BALANCER_API_BASE_URL = 'https://mindrouter-api.nkn.uidaho.edu';
    let gpuChartInstance = null;

    const fetchData = async () => {
        try {
            showLoader();
            const response = await fetch(`${BALANCER_API_BASE_URL}/collect-info`);
            const data = await response.json();
            hideLoader();

            updateServersList(data);
            updateGpuList(data);
            updateModelsList(data);
            updateGpuChart(data);
        } catch (error) {
            console.error('Error fetching data:', error);
            hideLoader();
        }
    };

    const showLoader = () => {
        const loader = document.createElement('div');
        loader.id = 'loader';
        loader.innerHTML = '<div class="spinner"></div>';
        document.body.appendChild(loader);
    };

    const hideLoader = () => {
        const loader = document.getElementById('loader');
        if (loader) {
            loader.remove();
        }
    };

    const updateServersList = (data) => {
        serversList.innerHTML = '';
        const serversGrid = document.createElement('div');
        serversGrid.className = 'servers-grid';
        
        data.forEach(node => {
            const serverDiv = document.createElement('div');
            serverDiv.className = 'server';
            const healthStatus = node.health ? 'Healthy' : 'Unhealthy';
            const healthColor = node.health ? 'green' : 'red';

            const hostInfo = node.gpu_info.length > 0 ? node.gpu_info[0].host : {};
            const hostDetails = `
                <p>Hostname: ${hostInfo.hostname || 'N/A'}</p>
                <p>CPU Load: ${hostInfo.cpu_load || 'N/A'}</p>
                <p>OS Type: ${hostInfo.os_type || 'N/A'}</p>
                <p>OS Version: ${hostInfo.os_version || 'N/A'}</p>
                <p>Memory Utilization: ${hostInfo.memory_utilization || 'N/A'}</p>
                <p>CPU Cores: ${hostInfo.cpu_cores || 'N/A'}</p>
                <p>Swap Utilization: ${hostInfo.swap_utilization || 'N/A'}</p>
            `;

            serverDiv.innerHTML = `<h3>Server: ${node.node}</h3>
                                   <p style="color:${healthColor}">${healthStatus}</p>
                                   ${hostDetails}`;
            serversGrid.appendChild(serverDiv);
        });
        
        serversList.appendChild(serversGrid);
    };

    const updateGpuList = (data) => {
        gpuList.innerHTML = '';
        data.forEach(node => {
            node.gpu_info.forEach(gpuDetails => {
                const nodeAlias = gpuDetails.host.hostname || node.node;
                const nodeDiv = document.createElement('div');
                nodeDiv.className = 'node';
                nodeDiv.innerHTML = `<h3>Node: ${nodeAlias}</h3>`;
                
                const gpusContainer = document.createElement('div');
                gpusContainer.className = 'gpus-container';

                gpuDetails.gpus.forEach(gpu => {
                    const gpuDiv = document.createElement('div');
                    gpuDiv.className = 'gpu';
                    const memoryAvailablePercentage = ((parseInt(gpu.memory_total) - parseInt(gpu.memory_used)) / parseInt(gpu.memory_total) * 100).toFixed(2);
                    
                    if (memoryAvailablePercentage >= 50) {
                        gpuDiv.classList.add('gpu-memory-high');
                    } else if (memoryAvailablePercentage >= 25) {
                        gpuDiv.classList.add('gpu-memory-medium');
                    } else {
                        gpuDiv.classList.add('gpu-memory-low');
                    }

                    gpuDiv.innerHTML = `
                        <p>GPU Name: ${gpu.name}</p>
                        <p>GPU Index: ${gpu.index}</p>
                        <p>Compute Utilization: ${gpu.utilization}%</p>
                        <p>Memory Used: ${gpu.memory_used}</p>
                        <p>Memory Available: ${parseInt(gpu.memory_total) - parseInt(gpu.memory_used)} (${memoryAvailablePercentage}%)</p>
                        <p>Temperature: ${gpu.temperature}°C</p>
                    `;
                    gpusContainer.appendChild(gpuDiv);
                });

                nodeDiv.appendChild(gpusContainer);
                gpuList.appendChild(nodeDiv);
            });
        });
    };

    const updateModelsList = (data) => {
        modelsList.innerHTML = '';
        data.forEach(node => {
            const nodeDiv = document.createElement('div');
            nodeDiv.className = 'node';
            nodeDiv.innerHTML = `<h3>Node: ${node.node}</h3>`;
            
            const modelsContainer = document.createElement('div');
            modelsContainer.className = 'models-container';

            node.ollama_info.forEach(ollamaInfo => {
                ollamaInfo.ollama_services.forEach(service => {
                    service.models_running.forEach(modelDetails => {
                        const modelDiv = document.createElement('div');
                        modelDiv.className = 'model';
                        let modelInfo = `<p><strong>Model:</strong> ${modelDetails.name}</p>`;
                        
                        if (modelDetails.size) {
                            modelInfo += `<p><strong>Size:</strong> ${modelDetails.size}</p>`;
                        }
                        if (modelDetails.processor) {
                            modelInfo += `<p><strong>Processor:</strong> ${modelDetails.processor}</p>`;
                        }

                        modelDiv.innerHTML = modelInfo;
                        modelsContainer.appendChild(modelDiv);
                    });
                });
            });

            nodeDiv.appendChild(modelsContainer);
            modelsList.appendChild(nodeDiv);
        });
    };

    const updateGpuChart = (data) => {
        const labels = [];
        const gpuUtilizationData = [];

        data.forEach(node => {
            node.gpu_info.forEach(gpuDetails => {
                gpuDetails.gpus.forEach(gpu => {
                    labels.push(`Node: ${gpuDetails.host.hostname}, GPU Index: ${gpu.index}`);
                    gpuUtilizationData.push(gpu.utilization);
                });
            });
        });

        if (gpuChartInstance) {
            gpuChartInstance.destroy();
        }

        gpuChartInstance = new Chart(gpuChartCtx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'GPU Utilization (%)',
                    data: gpuUtilizationData,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                    fill: false,
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    };

    fetchData();
    setInterval(fetchData, 60000); // Refresh every minute
});
