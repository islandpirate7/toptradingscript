/**
 * Index page specific JavaScript
 */
document.addEventListener('DOMContentLoaded', function() {
    // Toggle custom date range fields
    const useCustomDateRange = document.getElementById('useCustomDateRange');
    if (useCustomDateRange) {
        useCustomDateRange.addEventListener('change', function() {
            const customDateRangeFields = document.querySelector('.custom-date-range-fields');
            if (customDateRangeFields) {
                customDateRangeFields.style.display = this.checked ? 'block' : 'none';
            }
        });
    }
});
