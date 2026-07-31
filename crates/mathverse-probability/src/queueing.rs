//! Queueing theory: M/M/1, M/M/c, arrival processes, service time distributions, waiting times.

/// M/M/1 queue (Poisson arrivals, exponential service, single server).
pub struct MM1Queue {
    pub arrival_rate: f64,
    pub service_rate: f64,
}

impl MM1Queue {
    pub fn new(arrival_rate: f64, service_rate: f64) -> Result<Self, String> {
        if arrival_rate <= 0.0 || service_rate <= 0.0 {
            return Err("Rates must be positive".to_string());
        }
        if arrival_rate >= service_rate {
            return Err("System unstable: arrival rate >= service rate".to_string());
        }
        
        Ok(MM1Queue {
            arrival_rate,
            service_rate,
        })
    }

    /// Traffic intensity (utilization).
    pub fn utilization(&self) -> f64 {
        self.arrival_rate / self.service_rate
    }

    /// Average number of customers in system (Little's Law).
    pub fn average_number_in_system(&self) -> f64 {
        let rho = self.utilization();
        rho / (1.0 - rho)
    }

    /// Average number of customers in queue.
    pub fn average_number_in_queue(&self) -> f64 {
        let rho = self.utilization();
        rho * rho / (1.0 - rho)
    }

    /// Average time in system.
    pub fn average_time_in_system(&self) -> f64 {
        1.0 / (self.service_rate - self.arrival_rate)
    }

    /// Average waiting time in queue.
    pub fn average_waiting_time(&self) -> f64 {
        self.arrival_rate / (self.service_rate * (self.service_rate - self.arrival_rate))
    }

    /// Probability of n customers in system.
    pub fn probability_n_customers(&self, n: usize) -> f64 {
        let rho = self.utilization();
        (1.0 - rho) * rho.powi(n as i32)
    }

    /// Probability that system is empty.
    pub fn probability_empty(&self) -> f64 {
        1.0 - self.utilization()
    }

    /// Probability that wait exceeds t.
    pub fn probability_wait_exceeds(&self, t: f64) -> f64 {
        let rho = self.utilization();
        rho * (-(self.service_rate - self.arrival_rate) * t).exp()
    }
}

/// M/M/c queue (multiple servers).
pub struct MMCQueue {
    pub arrival_rate: f64,
    pub service_rate: f64,
    pub n_servers: usize,
}

impl MMCQueue {
    pub fn new(arrival_rate: f64, service_rate: f64, n_servers: usize) -> Result<Self, String> {
        if arrival_rate <= 0.0 || service_rate <= 0.0 || n_servers == 0 {
            return Err("Invalid parameters".to_string());
        }
        if arrival_rate >= n_servers as f64 * service_rate {
            return Err("System unstable".to_string());
        }
        
        Ok(MMCQueue {
            arrival_rate,
            service_rate,
            n_servers,
        })
    }

    /// Traffic intensity per server.
    pub fn utilization(&self) -> f64 {
        self.arrival_rate / (self.n_servers as f64 * self.service_rate)
    }

    /// Probability that system is empty (Erlang C formula).
    pub fn probability_empty(&self) -> f64 {
        let rho = self.arrival_rate / self.service_rate;
        let c = self.n_servers as f64;
        
        // Compute P0
        let mut sum = 0.0;
        for n in 0..self.n_servers {
            sum += rho.powi(n as i32) / (n as f64).gamma();
        }
        
        let last_term = rho.powi(self.n_servers as i32) / (self.n_servers as f64).gamma() 
            / (1.0 - rho / c);
        
        1.0 / (sum + last_term)
    }

    /// Probability that an arriving customer waits.
    pub fn probability_wait(&self) -> f64 {
        let rho = self.arrival_rate / self.service_rate;
        let c = self.n_servers as f64;
        let p0 = self.probability_empty();
        
        let numerator = rho.powi(self.n_servers as i32) / (self.n_servers as f64).gamma();
        let denominator = (1.0 - rho / c) * (c * c);
        
        p0 * numerator / denominator
    }

    /// Average number in queue.
    pub fn average_number_in_queue(&self) -> f64 {
        let c = self.n_servers as f64;
        let rho = self.arrival_rate / self.service_rate;
        let p_wait = self.probability_wait();
        
        p_wait * rho / (c - rho)
    }

    /// Average number in system.
    pub fn average_number_in_system(&self) -> f64 {
        self.average_number_in_queue() + self.arrival_rate / self.service_rate
    }

    /// Average waiting time.
    pub fn average_waiting_time(&self) -> f64 {
        self.average_number_in_queue() / self.arrival_rate
    }

    /// Average time in system.
    pub fn average_time_in_system(&self) -> f64 {
        self.average_waiting_time() + 1.0 / self.service_rate
    }
}

/// M/G/1 queue (general service time distribution).
pub struct MG1Queue {
    pub arrival_rate: f64,
    pub mean_service_time: f64,
    pub variance_service_time: f64,
}

impl MG1Queue {
    pub fn new(arrival_rate: f64, mean_service_time: f64, variance_service_time: f64) -> Result<Self, String> {
        if arrival_rate <= 0.0 || mean_service_time <= 0.0 {
            return Err("Invalid parameters".to_string());
        }
        if arrival_rate * mean_service_time >= 1.0 {
            return Err("System unstable".to_string());
        }
        
        Ok(MG1Queue {
            arrival_rate,
            mean_service_time,
            variance_service_time,
        })
    }

    /// Traffic intensity.
    pub fn utilization(&self) -> f64 {
        self.arrival_rate * self.mean_service_time
    }

    /// Pollaczek-Khinchine formula: average number in queue.
    pub fn average_number_in_queue(&self) -> f64 {
        let rho = self.utilization();
        let lambda = self.arrival_rate;
        let es = self.mean_service_time;
        let var_s = self.variance_service_time;
        
        (lambda * lambda * var_s + rho * rho) / (2.0 * (1.0 - rho))
    }

    /// Average number in system.
    pub fn average_number_in_system(&self) -> f64 {
        self.average_number_in_queue() + self.utilization()
    }

    /// Average waiting time.
    pub fn average_waiting_time(&self) -> f64 {
        self.average_number_in_queue() / self.arrival_rate
    }

    /// Average time in system.
    pub fn average_time_in_system(&self) -> f64 {
        self.average_waiting_time() + self.mean_service_time
    }
}

/// G/G/1 queue (general arrival and service).
pub struct GG1Queue {
    pub arrival_rate: f64,
    pub mean_arrival_time: f64,
    pub variance_arrival_time: f64,
    pub mean_service_time: f64,
    pub variance_service_time: f64,
}

impl GG1Queue {
    pub fn new(
        arrival_rate: f64,
        mean_arrival_time: f64,
        variance_arrival_time: f64,
        mean_service_time: f64,
        variance_service_time: f64,
    ) -> Result<Self, String> {
        if arrival_rate <= 0.0 || mean_arrival_time <= 0.0 || mean_service_time <= 0.0 {
            return Err("Invalid parameters".to_string());
        }
        if arrival_rate * mean_service_time >= 1.0 {
            return Err("System unstable".to_string());
        }
        
        Ok(GG1Queue {
            arrival_rate,
            mean_arrival_time,
            variance_arrival_time,
            mean_service_time,
            variance_service_time,
        })
    }

    /// Kingman's approximation for average waiting time.
    pub fn average_waiting_time_approx(&self) -> f64 {
        let rho = self.arrival_rate * self.mean_service_time;
        let ca_sq = self.variance_arrival_time / (self.mean_arrival_time * self.mean_arrival_time);
        let cs_sq = self.variance_service_time / (self.mean_service_time * self.mean_service_time);
        
        (rho / (1.0 - rho)) * (self.mean_service_time / 2.0) * (ca_sq + cs_sq)
    }
}

/// Birth-death process queue.
pub struct BirthDeathQueue {
    pub birth_rates: Vec<f64>,
    pub death_rates: Vec<f64>,
}

impl BirthDeathQueue {
    pub fn new(birth_rates: Vec<f64>, death_rates: Vec<f64>) -> Result<Self, String> {
        if birth_rates.len() != death_rates.len() {
            return Err("Birth and death rate vectors must have same length".to_string());
        }
        
        Ok(BirthDeathQueue {
            birth_rates,
            death_rates,
        })
    }

    /// Steady-state probabilities.
    pub fn steady_state_probabilities(&self) -> Result<Vec<f64>, String> {
        let n = self.birth_rates.len();
        let mut pi = vec![0.0; n + 1];
        
        // Compute pi[0]
        let mut product = 1.0;
        let mut sum = 1.0;
        
        for i in 1..=n {
            product *= self.birth_rates[i - 1] / self.death_rates[i];
            sum += product;
        }
        
        pi[0] = 1.0 / sum;
        
        // Compute remaining probabilities
        product = 1.0;
        for i in 1..=n {
            product *= self.birth_rates[i - 1] / self.death_rates[i];
            pi[i] = pi[0] * product;
        }
        
        Ok(pi)
    }

    /// Average number in system.
    pub fn average_number_in_system(&self) -> Result<f64, String> {
        let pi = self.steady_state_probabilities()?;
        let mut mean = 0.0;
        
        for (i, &p) in pi.iter().enumerate() {
            mean += i as f64 * p;
        }
        
        Ok(mean)
    }
}

/// Little's Law.
pub struct LittlesLaw;

impl LittlesLaw {
    /// L = λW: average number in system = arrival rate × average time in system.
    pub fn number_from_time(arrival_rate: f64, average_time: f64) -> f64 {
        arrival_rate * average_time
    }

    /// W = L/λ: average time in system = average number / arrival rate.
    pub fn time_from_number(average_number: f64, arrival_rate: f64) -> f64 {
        if arrival_rate > 0.0 {
            average_number / arrival_rate
        } else {
            f64::INFINITY
        }
    }

    /// λ = L/W: arrival rate = average number / average time.
    pub fn rate_from_number_time(average_number: f64, average_time: f64) -> f64 {
        if average_time > 0.0 {
            average_number / average_time
        } else {
            0.0
        }
    }
}

/// Queueing network analysis.
pub struct QueueingNetwork;

impl QueueingNetwork {
    /// Open Jackson network (simplified).
    pub fn jackson_network(
        arrival_rates: Vec<f64>,
        service_rates: Vec<f64>,
        routing_matrix: Vec<Vec<f64>>,
    ) -> Result<Vec<f64>, String> {
        let n = arrival_rates.len();
        if service_rates.len() != n || routing_matrix.len() != n {
            return Err("Dimension mismatch".to_string());
        }
        
        // Solve traffic equations: λ = γ + Pλ
        let mut lambda = arrival_rates.clone();
        
        for _ in 0..1000 {
            let mut new_lambda = arrival_rates.clone();
            
            for i in 0..n {
                for j in 0..n {
                    new_lambda[i] += routing_matrix[j][i] * lambda[j];
                }
            }
            
            // Check convergence
            let diff: f64 = lambda.iter().zip(new_lambda.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            
            lambda = new_lambda;
            
            if diff < 1e-10 {
                break;
            }
        }
        
        Ok(lambda)
    }

    /// Closed Jackson network (simplified).
    pub fn closed_jackson_network(
        service_rates: Vec<f64>,
        routing_matrix: Vec<Vec<f64>>,
        n_customers: usize,
    ) -> Result<Vec<f64>, String> {
        let n = service_rates.len();
        if routing_matrix.len() != n {
            return Err("Dimension mismatch".to_string());
        }
        
        // Simplified: assume equal visitation
        let visitation = vec![1.0 / n as f64; n];
        let throughput = n_customers as f64 / (visitation.iter()
            .zip(service_rates.iter())
            .map(|(&v, &s)| v / s)
            .sum::<f64>());
        
        let lambda: Vec<f64> = visitation.iter().map(|&v| v * throughput).collect();
        Ok(lambda)
    }
}

/// Queue discipline effects.
pub enum QueueDiscipline {
    FIFO,  // First-In-First-Out
    LIFO,  // Last-In-First-Out
    SIRO,  // Service-In-Random-Order
    Priority,  // Priority queue
}

impl QueueDiscipline {
    /// Effect on average waiting time (relative to FIFO).
    pub fn waiting_time_factor(&self) -> f64 {
        match self {
            QueueDiscipline::FIFO => 1.0,
            QueueDiscipline::LIFO => 1.0,  // Same average, different distribution
            QueueDiscipline::SIRO => 1.0,  // Same average
            QueueDiscipline::Priority => 0.5,  // Approximate for high priority
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mm1_queue() {
        let queue = MM1Queue::new(2.0, 5.0).unwrap();
        assert!((queue.utilization() - 0.4).abs() < 1e-10);
        
        let l = queue.average_number_in_system();
        assert!((l - 0.4 / 0.6).abs() < 1e-10);
        
        let w = queue.average_time_in_system();
        assert!((w - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mmc_queue() {
        let queue = MMCQueue::new(4.0, 2.0, 3).unwrap();
        assert!(queue.utilization() < 1.0);
        
        let l = queue.average_number_in_system();
        assert!(l > 0.0);
    }

    #[test]
    fn test_mg1_queue() {
        let queue = MG1Queue::new(2.0, 0.3, 0.1).unwrap();
        assert!(queue.utilization() < 1.0);
        
        let lq = queue.average_number_in_queue();
        assert!(lq > 0.0);
    }

    #[test]
    fn test_littles_law() {
        let lambda = 10.0;
        let w = 5.0;
        let l = LittlesLaw::number_from_time(lambda, w);
        assert!((l - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_birth_death_queue() {
        let birth_rates = vec![2.0, 2.0, 2.0];
        let death_rates = vec![3.0, 3.0, 3.0];
        let queue = BirthDeathQueue::new(birth_rates, death_rates).unwrap();
        
        let pi = queue.steady_state_probabilities().unwrap();
        let sum: f64 = pi.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
