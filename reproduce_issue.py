import time
from playwright.sync_api import sync_playwright


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Capture console logs
        page.on("console", lambda msg: print(f"Browser Console: {msg.text}"))
        page.on("pageerror", lambda exc: print(f"Browser Error: {exc}"))

        print("Navigating to WebUI...")
        page.goto("http://localhost:5173")

        # Wait for loading
        page.wait_for_load_state("networkidle")
        print("Page loaded.")

        # Take initial screenshot
        page.screenshot(path="screenshot_initial.png")

        # Look for the Task dropdown.
        # In the code Sidebar.tsx, the button has text "Select a task..." or the current task name.
        # We can find the button that precedes the "Result" label or just search by text.

        # Click task dropdown trigger
        print("Opening task dropdown...")
        try:
            # Try to find the button that toggles the task list
            # It has text "Select a task..." initially
            page.get_by_text("Select a task...").click()
        except:
            # Maybe a task is already selected?
            print(
                "Could not find 'Select a task...', checking if a task is pre-selected."
            )
            # This might happen if state is persisted or default.

        # Wait for dropdown to appear
        time.sleep(1)

        # Try to click "squeeze_hnsw"
        try:
            print("Selecting 'squeeze_hnsw'...")
            page.get_by_text("squeeze_hnsw").click()
        except Exception as e:
            print(f"Could not find 'squeeze_hnsw' option: {e}")
            # If not found, maybe try any available task?
            # Let's list what we see
            page.screenshot(path="screenshot_dropdown.png")
            browser.close()
            return

        # Wait for results to load and select the first one
        time.sleep(1)
        print("Selecting first result...")
        try:
            # Click the "Select a result..." button
            page.get_by_text("Select a result...").click()
            time.sleep(0.5)
            # Click the first result button in the dropdown (assuming it exists)
            # The result buttons are in the dropdown container
            # We can just pick the first one that looks like a result
            # The sidebar logic renders buttons in the dropdown.
            # Let's just try to find any button in the result dropdown.
            # Or we can wait for the database to load if it auto-selects?
            # The code says: "Auto-select first result if available" in handleTaskSelect
            # So maybe we don't need to manually select a result if the task switch triggers it.
            pass
        except:
            print("Could not interact with result dropdown, maybe already selected.")

        # Wait for the table to populate
        print("Waiting for programs table...")
        try:
            # Wait for at least one row
            page.wait_for_selector(
                "tr.model-cell", timeout=5000
            )  # Attempting to find a row
        except:
            # The table rows have class 'model-cell' on one td, but the row itself doesn't have a specific class other than maybe selected/incorrect
            # Let's look for a td.
            try:
                page.wait_for_selector("table.programs-table tbody tr", timeout=5000)
            except Exception as e:
                print(f"Table did not populate: {e}")
                page.screenshot(path="screenshot_table_fail.png")
                browser.close()
                return

        print("Table populated. Clicking first program row...")
        # Click the first row
        rows = page.locator("table.programs-table tbody tr")
        if rows.count() > 0:
            rows.first.click()
            print("Clicked first row.")
        else:
            print("No rows found.")

        # Wait a bit for the "White Screen" or Code View
        time.sleep(2)

        # Screenshot result
        page.screenshot(path="screenshot_after_click.png")
        print("Screenshot saved to 'screenshot_after_click.png'")

        # Check if Error Boundary triggered
        if page.get_by_text("Something went wrong").is_visible():
            print("!!! Error Boundary Caught an Error !!!")
            # Extract error text
            error_text = page.locator("pre").text_content()
            print(f"Error content: {error_text}")

        browser.close()


if __name__ == "__main__":
    run()
