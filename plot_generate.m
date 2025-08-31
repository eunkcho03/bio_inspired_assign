% Read data from Excel
data = readtable('metrics_copy.xlsx');

% Extract columns
step = data.step;
avg_return = data.avg_return;
success_rate = data.success_rate;

%% Plot 1: Step vs Avg Return
figure;
set(gcf, 'Position', [100, 100, 500, 250]);
plot(step, avg_return, 'b-', 'LineWidth', 1.5);
xlabel('Step');
ylabel('Average Return');
%title('Step vs Average Return');
grid on;
saveas(gcf, 'avg_return_plot.png');  % Save as image

%% Plot 2: Step vs Success Rate
figure;
set(gcf, 'Position', [100, 100, 500, 250]);
plot(step, success_rate, 'r-', 'LineWidth', 1.5);
xlabel('Step');
ylabel('Success Rate');
%title('Step vs Success Rate');
grid on;
saveas(gcf, 'success_rate_plot.png');  % Save as image

%% Save processed data back to Excel
T = table(step, avg_return, success_rate);
writetable(T, 'processed_metrics.xlsx');
