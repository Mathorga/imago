#include <imago/imago.h>
#include <SFML/Graphics.hpp>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

float randomFloat(float min, float max) {
    float random = ((float)rand()) / (float)RAND_MAX;

    float range = max - min;
    return (random * range) + min;
}

void initPositions(corticolumn column, float* xNeuronPositions, float* yNeuronPositions) {
    for (uint32_t i = 0; i < column.neurons_count; i++) {
        xNeuronPositions[i] = randomFloat(0, 1);
        yNeuronPositions[i] = randomFloat(0, 1);
    }
}

int main(int argc, char **argv) {
    int neuronsCount = 100;
    int synapsesDensity = 2;

    // Input handling.
    switch (argc) {
        case 2:
            neuronsCount = atoi(argv[1]);
            break;
        case 3:
            neuronsCount = atoi(argv[1]);
            synapsesDensity = atoi(argv[2]);
            break;
        default:
            break;
    }

    sf::VideoMode desktopMode = sf::VideoMode::getDesktopMode();


    srand(time(0));

    // Create network model.
    corticolumn column;
    dccol_init(&column, neuronsCount, synapsesDensity);

    float* xNeuronPositions = (float*) malloc(column.neurons_count * sizeof(float));
    float* yNeuronPositions = (float*) malloc(column.neurons_count * sizeof(float));

    initPositions(column, xNeuronPositions, yNeuronPositions);
    
    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;

    // create the window
    sf::RenderWindow window(desktopMode, "Imago", sf::Style::Fullscreen, settings);

    // Run the program as long as the window is open
    while (window.isOpen()) {
        usleep(10000);

        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event)) {
            // "close requested" event: we close the window
            switch(event.type) {
                case sf::Event::Closed:
                    window.close();
                    break;
                case sf::Event::KeyReleased:
                    if (event.key.code == sf::Keyboard::Space) {
                        initPositions(column, xNeuronPositions, yNeuronPositions);
                    }
                    break;
                default:
                    break;
            }
        }

        // Clear the window with black color
        window.clear(sf::Color(31, 31, 31, 255));

        uint32_t input_neurons[] = {0, 1, 2, 3};
        if (randomFloat(0, 1) < 0.4f) {
            ccol_feed(&column, input_neurons, 4, DEFAULT_VALUE);
        }
        ccol_tick(&column);

        // Draw neurons.
        for (uint32_t i = 0; i < column.neurons_count; i++) {
            sf::CircleShape neuronSpot;
            float radius = 5.0f;
            neuronSpot.setRadius(radius);
            neuronSpot.setFillColor(sf::Color(64, 64, (sf::Uint8) column.neurons[i].value, 255));
            
            neuronSpot.setPosition(xNeuronPositions[i] * desktopMode.width, yNeuronPositions[i] * desktopMode.height);

            // Center the spot.
            neuronSpot.setOrigin(radius, radius);
            window.draw(neuronSpot);
        }

        // Draw synapses.
        for (uint32_t i = 0; i < column.synapses_count; i++) {
            sf::Vertex line[] = {
                sf::Vertex(
                    {xNeuronPositions[column.synapses[i].input_neuron] * desktopMode.width, yNeuronPositions[column.synapses[i].input_neuron] * desktopMode.height},
                    sf::Color(255, 127, 31, 31)),
                sf::Vertex(
                    {xNeuronPositions[column.synapses[i].output_neuron] * desktopMode.width, yNeuronPositions[column.synapses[i].output_neuron] * desktopMode.height},
                    sf::Color(31, 127, 255, 31))
            };

            window.draw(line, 2, sf::Lines);
        }

        // Draw spikes count.
        sf::Font font;
        if (!font.loadFromFile("res/JetBrainsMono.ttf")) {
            printf("Font not loaded\n");
        }
        sf::Text infoText;
        infoText.setPosition(20.0f, 20.0f);
        infoText.setString("Spikes count: " + std::to_string(column.spikes_count));
        infoText.setFont(font);
        infoText.setCharacterSize(24);
        infoText.setFillColor(sf::Color::White);
        window.draw(infoText);

        // Draw spikes.
        for (uint32_t i = 0; i < column.spikes_count; i++) {
            sf::CircleShape spikeSpot;
            float radius = 2.0f;
            spikeSpot.setRadius(radius);
            spikeSpot.setFillColor(sf::Color(255, 255, 255, 31));

            synapse* referenceSynapse = &(column.synapses[column.spikes[i].synapse]);
            uint32_t startingNeuron = referenceSynapse->input_neuron;
            uint32_t endingNeuron = referenceSynapse->output_neuron;

            float xDist = xNeuronPositions[endingNeuron] * desktopMode.width - xNeuronPositions[startingNeuron] * desktopMode.width;
            float yDist = yNeuronPositions[endingNeuron] * desktopMode.height - yNeuronPositions[startingNeuron] * desktopMode.height;

            float progress = ((float) column.spikes[i].progress) / ((float) referenceSynapse->propagation_time);

            float xSpikePosition = xNeuronPositions[startingNeuron] * desktopMode.width + (progress * (xDist));
            float ySpikePosition = yNeuronPositions[startingNeuron] * desktopMode.height + (progress * (yDist));
            
            spikeSpot.setPosition(xSpikePosition, ySpikePosition);

            // Center the spot.
            spikeSpot.setOrigin(radius, radius);
            window.draw(spikeSpot);
        }

        // End the current frame.
        window.display();
    }

    return 0;
}