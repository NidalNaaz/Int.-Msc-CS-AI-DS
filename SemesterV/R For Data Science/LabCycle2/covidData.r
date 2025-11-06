library(readr)
library(dplyr)
library(ggplot2)
library(forecast)

covid_data <- read_csv(
  "~/Desktop/R Lab/LC2/time-series-19-covid-combined.csv",
  show_col_types = FALSE
)

covid_data$Date <- as.Date(covid_data$Date, format = "%Y-%m-%d")

global_data <- covid_data %>%
  group_by(Date) %>%
  summarise(Confirmed = sum(Confirmed, na.rm = TRUE)) %>%
  arrange(Date)

print(head(global_data))
cat("Total rows in global_data:", nrow(global_data), "\n")

plot_ts <- ggplot(global_data, aes(x = Date, y = Confirmed)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_smooth(method = "loess", se = FALSE, color = "red", linewidth = 1) +
  labs(title = "Global COVID-19 Confirmed Cases with Trend Line",
       x = "Date",
       y = "Number of Confirmed Cases") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(plot_ts)

ts_data <- ts(global_data$Confirmed, frequency = 365, start = c(2020, 1))

fit <- auto.arima(ts_data)
forecasted_values <- forecast(fit, h = 30)

plot_forecast <- autoplot(forecasted_values) +
  labs(title = "Forecast of Global COVID-19 Confirmed Cases",
       x = "Date",
       y = "Number of Confirmed Cases") +
  theme_minimal()

print(plot_forecast)
