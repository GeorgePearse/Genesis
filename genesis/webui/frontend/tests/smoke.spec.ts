import { expect, test } from '@playwright/test';

test('loads the shell and empty state without a database', async ({ page }) => {
  await page.route('http://localhost:8000/list_databases', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([]),
    });
  });

  await page.route('http://localhost:8000/get_programs**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([]),
    });
  });

  await page.goto('/');

  await expect(
    page.getByRole('heading', { name: 'Genesis', exact: true })
  ).toBeVisible();
  await expect(
    page.getByRole('button', { name: 'Search', exact: true })
  ).toBeVisible();
  await expect(
    page.getByRole('heading', { name: 'No Database Selected', exact: true })
  ).toBeVisible();

  await page.getByRole('button', { name: 'Search', exact: true }).click();

  await expect(page.getByRole('option', { name: 'Tree View' })).toBeVisible();
});
