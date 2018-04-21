/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    default_random_engine random_generator;
    num_particles = 100;
    is_initialized = true;

    // Create a normal (Gaussian) distribution for x, y and theta
    normal_distribution<double> dist_x(0, std[0]);
    normal_distribution<double> dist_y(0, std[1]);
    normal_distribution<double> dist_theta(0, std[2]);

    // Fill a number of particles and weights into the vectors
    // Init particles with Gaussian noise and weight 1
    for (int i = 0; i < num_particles; ++i) {
        Particle p;
        p.id = i;
        p.x = x + dist_x(random_generator);
        p.y = y + dist_y(random_generator),
                p.theta = theta + dist_theta(random_generator);
        p.weight = 1.0;

        particles.push_back(p);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine random_generator;

    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    if (abs(yaw_rate) < 1e-5) {
        // Vehicle is moving straight 
        for (int i = 0; i < num_particles; ++i) {

            double theta = particles[i].theta;
            particles[i].x += velocity * delta_t * cos(theta) + dist_x(random_generator);
            particles[i].y += velocity * delta_t * sin(theta) + dist_y(random_generator);
            particles[i].theta = theta + dist_theta(random_generator);
        }

    } else {
        // Vehicle is turning 
        for (int i = 0; i < num_particles; ++i) {

            auto theta = particles[i].theta;
            auto theta_predicted = theta + delta_t*yaw_rate;

            particles[i].x += velocity / yaw_rate * (sin(theta_predicted) - sin(theta)) + dist_x(random_generator);
            particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta_predicted)) + dist_y(random_generator);
            particles[i].theta = theta_predicted + dist_theta(random_generator);
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.

    for (auto& observation : observations) {

        double distance = std::numeric_limits<float>::max();

        for (const auto& pred : predicted) {

            double distance_obs_pred = dist(observation.x, observation.y, pred.x, pred.y);
            if (distance_obs_pred < distance) {
                observation.id = pred.id;
                distance = distance_obs_pred;
            }
        }
    }

}

double ParticleFilter::calculateParticleWeight(std::vector<LandmarkObs> predicted_obs,
        std::vector<LandmarkObs> observations_map,
        double std_landmark[]) {

    double weight = 1.0;
    double mu_x, mu_y;

    auto sig_x = std_landmark[0];
    auto sig_y = std_landmark[1];

    for (auto& observation : observations_map) {
        auto x_obs = observation.x;
        auto y_obs = observation.y;

        for (const auto& pred : predicted_obs) {
            if (pred.id == observation.id) {
                mu_x = pred.x;
                mu_y = pred.y;
                break;
            }
        }

        // calculate normalization term
        double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

        // calculate exponent
        double exponent = pow((x_obs - mu_x), 2) / (2 * pow(sig_x, 2)) +
                pow((y_obs - mu_y), 2) / (2 * pow(sig_y, 2));

        // calculate weight using normalization terms and exponent
        weight *= gauss_norm * exp(-exponent);
    }

    return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

   
    weights.clear();

    for (auto& particle : particles) {

        vector<LandmarkObs> observations_map_coordinate;
        vector<LandmarkObs> predicted_observations;

        // Transform observations from Sensor-frame to Map-frame
        auto x_p = particle.x;
        auto y_p = particle.y;
        auto theta = particle.theta;

        observations_map_coordinate.clear();
        LandmarkObs lobs;
        for (const auto& observation : observations) {

            lobs.id = -1;
            auto x = observation.x;
            auto y = observation.y;

            lobs.x = x_p + x * cos(theta) + y*-sin(theta);
            lobs.y = y_p + x * sin(theta) + y * cos(theta);
            observations_map_coordinate.push_back(lobs);
        }

        // Get all landmarks within the sensor range (in Map-frame)
        for (const auto& landmark : map_landmarks.landmark_list) {

            if (dist(landmark.x_f, landmark.y_f, particle.x, particle.y) <= sensor_range) {
                lobs.id = landmark.id_i;
                lobs.x = landmark.x_f;
                lobs.y = landmark.y_f;

                predicted_observations.push_back(lobs);
            }
        }

        dataAssociation(predicted_observations, observations_map_coordinate);
        SetAssociations(particle, observations_map_coordinate);

        // Calculate particle weights
        weights.push_back(calculateParticleWeight(predicted_observations, observations_map_coordinate, std_landmark));
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine random_generator;
    discrete_distribution<int> dist(weights.begin(), weights.end());

    vector<Particle> particles_resampled;
    for (uint i = 0; i < particles.size(); ++i) {
        particles_resampled.push_back(particles[dist(random_generator)]);
    }

    particles = particles_resampled;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<LandmarkObs>& lobs) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    for (const auto& obs : lobs) {
        particle.associations.push_back(obs.id);
        particle.sense_x.push_back(obs.x);
        particle.sense_y.push_back(obs.y);
    }
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}
