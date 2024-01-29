function do_figures_black (gca, do_all_lines_white)

    set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'GridColor', 'w'); % Set the current axes properties
    set(gcf, 'Color', 'k'); % Set the current figure background
    if do_all_lines_white == 1
        set(findall(gca, 'Type', 'Line'), 'Color', 'w'); % Set line plot colors to white
    else

    end
    box off
    % Set the ticks direction to outside
    set(gca, 'TickDir', 'out', 'TickLength', [0.02, 0.02]); % Adjust the tick length as desired
end